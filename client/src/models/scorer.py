"""
Zero-shot scorer module for essay scoring.

Implements deterministic (greedy) score extraction from LLM outputs.
"""

import re
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel, get_peft_model, LoraConfig

from .prompts import ScoringPromptBuilder, create_prompt_builder

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Container for a single scoring result."""
    essay_id: int
    y_true: Optional[int]
    y_hat_greedy: Optional[int]
    parsed_ok: bool
    raw_greedy_text: str
    y_hat_argmax: Optional[int] = None
    y_tilde: Optional[float] = None
    y_tilde_round: Optional[int] = None
    p_valid_sum: Optional[float] = None
    p_invalid: Optional[float] = None
    logp_by_score: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            'essay_id': self.essay_id,
            'y_true': self.y_true,
            'y_hat_greedy': self.y_hat_greedy,
            'parsed_ok': self.parsed_ok,
            'raw_greedy_text': self.raw_greedy_text,
            'y_hat_argmax': self.y_hat_argmax,
            'y_tilde': self.y_tilde,
            'y_tilde_round': self.y_tilde_round,
            'p_valid_sum': self.p_valid_sum,
            'p_invalid': self.p_invalid,
        }


class NewlineStoppingCriteria(StoppingCriteria):
    """Stopping criteria that stops on newline token."""

    def __init__(self, tokenizer: PreTrainedTokenizer, stop_token: str = "\n"):
        self.tokenizer = tokenizer
        self.stop_token = stop_token
        # Find newline token IDs
        self.stop_token_ids = self._get_stop_token_ids()

    def _get_stop_token_ids(self) -> List[int]:
        """Get token IDs that start with or are newline."""
        stop_ids = []
        for token_id in range(self.tokenizer.vocab_size):
            try:
                decoded = self.tokenizer.decode([token_id])
                if decoded.startswith(self.stop_token) or decoded == self.stop_token:
                    stop_ids.append(token_id)
            except Exception:
                continue
        return stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] == 0:
            return False
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_token_ids


class ConstrainedScoreLogitsProcessor(LogitsProcessor):
    """
    Logits processor that constrains output to valid score tokens only.

    Allows only:
    - First token: space, digit, or boundary tokens
    - Continuation after digits: digit or newline tokens
    - Continuation before digits: digit tokens only

    This ensures the model can only generate valid score outputs
    (digits optionally preceded by space and terminated by newline).
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, input_length: int):
        """
        Initialize the processor.

        Args:
            tokenizer: The tokenizer to use for token classification
            input_length: Length of input tokens (to know when generation starts)
        """
        self.tokenizer = tokenizer
        self.input_length = input_length

        # Build token sets
        self._space_tokens: set = set()
        self._digit_tokens: set = set()
        self._newline_tokens: set = set()
        self._boundary_tokens: set = set()
        self._build_token_sets()

    def _build_token_sets(self):
        """Pre-compute sets of valid tokens."""
        vocab_size = self.tokenizer.vocab_size

        for token_id in range(vocab_size):
            try:
                decoded = self.tokenizer.decode([token_id])

                # Boundary tokens: SentencePiece word boundary markers
                if decoded == "":
                    self._boundary_tokens.add(token_id)
                    continue

                # Space tokens: whitespace but NOT newline
                if decoded.strip() == "" and "\n" not in decoded:
                    self._space_tokens.add(token_id)

                # Digit tokens: tokens containing only digits (possibly with leading space)
                stripped = decoded.lstrip()
                if stripped and all(c.isdigit() for c in stripped):
                    self._digit_tokens.add(token_id)

                # Newline tokens: tokens containing newline
                if "\n" in decoded:
                    self._newline_tokens.add(token_id)

            except Exception:
                continue

    def _get_valid_first_tokens(self) -> set:
        """Get tokens valid as first generated token."""
        return self._space_tokens | self._digit_tokens | self._boundary_tokens

    def _get_valid_continuation_tokens(self, has_digits: bool) -> set:
        """Get tokens valid as continuation."""
        if has_digits:
            # After digits: more digits or newline
            return self._digit_tokens | self._newline_tokens
        else:
            # Before digits (after space/boundary): only digits
            return self._digit_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to mask invalid tokens.

        Args:
            input_ids: Current sequence of token IDs
            scores: Logits for next token

        Returns:
            Modified logits with invalid tokens masked to -inf
        """
        # Calculate how many tokens have been generated
        generated_length = input_ids.shape[1] - self.input_length

        if generated_length == 0:
            # First generated token
            valid_tokens = self._get_valid_first_tokens()
        else:
            # Check if we have digits in the generated portion
            generated_ids = input_ids[0, self.input_length:].tolist()
            generated_text = self.tokenizer.decode(generated_ids)
            has_digits = any(c.isdigit() for c in generated_text)
            valid_tokens = self._get_valid_continuation_tokens(has_digits)

        # Create mask: True for tokens to KEEP
        mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        for token_id in valid_tokens:
            if token_id < scores.shape[-1]:
                mask[token_id] = True

        # Set invalid tokens to -inf
        scores = scores.clone()
        scores[:, ~mask] = float('-inf')

        return scores


class ZeroShotScorer:
    """
    Zero-shot essay scorer using LLM with greedy decoding.

    Supports:
    - Loading base models and LoRA adapters
    - Greedy decoding for deterministic score extraction
    - Score parsing and validation
    """

    def __init__(
        self,
        model_name: str,
        y_min: int,
        y_max: int,
        prompt_id: int = 1,
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_generic_rubric: bool = True,
        output_prefix: str = "The score of this essay: ",
    ):
        """
        Initialize the scorer.

        Args:
            model_name: HuggingFace model name or path
            y_min: Minimum allowed score
            y_max: Maximum allowed score
            prompt_id: ASAP prompt ID
            device: Device to use
            dtype: Data type for model
            use_generic_rubric: Whether to use generic rubric
            output_prefix: Output prefix for score
        """
        self.model_name = model_name
        self.y_min = y_min
        self.y_max = y_max
        self.prompt_id = prompt_id
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.output_prefix = output_prefix

        # Initialize components
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.prompt_builder: Optional[ScoringPromptBuilder] = None
        self.stopping_criteria: Optional[StoppingCriteriaList] = None
        self._adapter_loaded = False

        # Create prompt builder
        self.prompt_builder = create_prompt_builder(
            prompt_id=prompt_id,
            y_min=y_min,
            y_max=y_max,
            use_generic_rubric=use_generic_rubric,
            output_prefix=output_prefix,
        )

    def load_model(self, adapter_path: Optional[str] = None):
        """
        Load the model and optionally a LoRA adapter.

        Args:
            adapter_path: Path to LoRA adapter (None for base model)
        """
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        # Load adapter if provided
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self._adapter_loaded = True
        else:
            self._adapter_loaded = False

        self.model.eval()

        # Setup stopping criteria
        self.stopping_criteria = StoppingCriteriaList([
            NewlineStoppingCriteria(self.tokenizer)
        ])

        logger.info("Model loaded successfully")

    def load_adapter(self, adapter_path: str):
        """
        Load or switch LoRA adapter.

        Args:
            adapter_path: Path to LoRA adapter
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._adapter_loaded:
            # Unload existing adapter first
            self.model = self.model.base_model.model

        if Path(adapter_path).exists():
            logger.info(f"Loading adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self._adapter_loaded = True
            self.model.eval()
        else:
            logger.warning(f"Adapter path not found: {adapter_path}")
            self._adapter_loaded = False

    def unload_adapter(self):
        """Unload current adapter, reverting to base model."""
        if self.model is None:
            return

        if self._adapter_loaded:
            self.model = self.model.base_model.model
            self._adapter_loaded = False
            self.model.eval()
            logger.info("Adapter unloaded, using base model")

    def set_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Set model and tokenizer from external source.

        This allows sharing a single model instance between trainer and scorer.

        Args:
            model: Pre-trained model (with or without LoRA)
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self._adapter_loaded = hasattr(model, 'peft_config')

        # Setup stopping criteria
        self.stopping_criteria = StoppingCriteriaList([
            NewlineStoppingCriteria(self.tokenizer)
        ])

        logger.info("Model set from external source")

    def _prepare_input(self, essay_text: str) -> Dict[str, torch.Tensor]:
        """Prepare model input from essay text."""
        messages = self.prompt_builder.to_messages(essay_text, use_prefill=True)

        # Apply chat template
        # Note: enable_thinking=False is passed for all models.
        # Models that don't support this parameter simply ignore it.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )

        # Note: We do NOT force trailing space here.
        # The logit extractor handles leading space tokens in the exploration,
        # which is more principled as it follows the chat template's design.

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def _parse_score(self, text: str) -> Tuple[Optional[int], bool]:
        """
        Parse score from generated text.

        Args:
            text: Generated text (after output prefix)

        Returns:
            Tuple of (parsed_score, success)
        """
        # Extract first integer from text
        match = re.match(r'^\s*([0-9]+)', text)
        if match:
            score = int(match.group(1))
            # Clamp to valid range
            score = max(self.y_min, min(self.y_max, score))
            return score, True
        return None, False

    @torch.inference_mode()  # Faster than no_grad()
    def score_essay_greedy(
        self,
        essay_text: str,
        max_new_tokens: int = 4,  # Reduced: scores are 1-2 digits
    ) -> Tuple[Optional[int], bool, str]:
        """
        Score an essay using constrained greedy decoding.

        Uses ConstrainedScoreLogitsProcessor to restrict output to valid
        score tokens only (digits, spaces, newlines), ensuring parseable output.

        Args:
            essay_text: Essay text to score
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Tuple of (score, parsed_ok, raw_text)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._prepare_input(essay_text)
        input_length = inputs['input_ids'].shape[1]

        # Create constrained logits processor
        logits_processor = LogitsProcessorList([
            ConstrainedScoreLogitsProcessor(self.tokenizer, input_length)
        ])

        # Generate with constrained decoding
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=self.stopping_criteria,
            logits_processor=logits_processor,
            use_cache=True,  # Enable KV cache
        )

        # Decode generated tokens only
        generated_tokens = outputs[0, input_length:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Parse score
        score, parsed_ok = self._parse_score(raw_text)

        return score, parsed_ok, raw_text

    def score_essay(
        self,
        essay_id: int,
        essay_text: str,
        y_true: Optional[int] = None,
        max_new_tokens: int = 12,
        compute_logit_expected: bool = False,
        logit_extractor=None,
    ) -> ScoringResult:
        """
        Score a single essay with full result.

        Args:
            essay_id: Essay identifier
            essay_text: Essay text
            y_true: Ground truth score (optional)
            max_new_tokens: Maximum tokens for greedy decoding
            compute_logit_expected: Whether to compute logit-based expected value
            logit_extractor: LogitExpectedValueExtractor instance

        Returns:
            ScoringResult with all scoring information
        """
        # Check if we can use unified extraction (1 forward pass for both)
        use_unified = (
            compute_logit_expected and
            logit_extractor is not None and
            hasattr(logit_extractor, 'extract_unified')
        )

        if use_unified:
            # Single forward pass for both greedy and logit extraction
            inputs = self._prepare_input(essay_text)
            logit_result = logit_extractor.extract_unified(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
            )

            result = ScoringResult(
                essay_id=essay_id,
                y_true=y_true,
                y_hat_greedy=logit_result.y_hat_greedy,
                parsed_ok=(logit_result.y_hat_greedy is not None),
                raw_greedy_text=logit_result.greedy_decoded,
                y_hat_argmax=logit_result.y_hat_argmax,
                y_tilde=logit_result.y_tilde,
                y_tilde_round=logit_result.y_tilde_round,
                p_valid_sum=logit_result.p_valid_sum,
                p_invalid=logit_result.p_invalid,
                logp_by_score=logit_result.logp_by_score,
            )
        else:
            # Original approach: separate greedy decoding and logit extraction
            # Greedy decoding
            y_hat_greedy, parsed_ok, raw_text = self.score_essay_greedy(
                essay_text, max_new_tokens
            )

            result = ScoringResult(
                essay_id=essay_id,
                y_true=y_true,
                y_hat_greedy=y_hat_greedy,
                parsed_ok=parsed_ok,
                raw_greedy_text=raw_text,
            )

            # Logit-based expected value extraction
            if compute_logit_expected and logit_extractor is not None:
                inputs = self._prepare_input(essay_text)
                logit_result = logit_extractor.extract(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                )
                result.y_hat_argmax = logit_result.y_hat_argmax
                result.y_tilde = logit_result.y_tilde
                result.y_tilde_round = logit_result.y_tilde_round
                result.p_valid_sum = logit_result.p_valid_sum
                result.p_invalid = logit_result.p_invalid
                result.logp_by_score = logit_result.logp_by_score

                # Use argmax as fallback if greedy parsing failed
                if not parsed_ok and result.y_hat_argmax is not None:
                    result.y_hat_greedy = result.y_hat_argmax
                    result.parsed_ok = True

        return result

    def score_essays(
        self,
        essays: List[Dict],
        max_new_tokens: int = 12,
        compute_logit_expected: bool = False,
        logit_extractor=None,
        show_progress: bool = True,
    ) -> List[ScoringResult]:
        """
        Score multiple essays.

        Args:
            essays: List of dicts with 'essay_id', 'essay_text', and optionally 'score'
            max_new_tokens: Maximum tokens for greedy decoding
            compute_logit_expected: Whether to compute logit-based expected value
            logit_extractor: LogitExpectedValueExtractor instance
            show_progress: Whether to show progress bar

        Returns:
            List of ScoringResult objects
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(essays, desc="Scoring essays")
            except ImportError:
                iterator = essays
        else:
            iterator = essays

        for essay in iterator:
            result = self.score_essay(
                essay_id=essay['essay_id'],
                essay_text=essay['essay_text'],
                y_true=essay.get('score'),
                max_new_tokens=max_new_tokens,
                compute_logit_expected=compute_logit_expected,
                logit_extractor=logit_extractor,
            )
            results.append(result)

        return results

    def get_input_for_logit_extraction(self, essay_text: str) -> Dict[str, torch.Tensor]:
        """
        Get prepared input for logit extraction.

        Args:
            essay_text: Essay text

        Returns:
            Dictionary with input tensors
        """
        return self._prepare_input(essay_text)


def load_scorer(
    model_name: str,
    prompt_id: int,
    y_min: int,
    y_max: int,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> ZeroShotScorer:
    """
    Convenience function to create and load a scorer.

    Args:
        model_name: HuggingFace model name
        prompt_id: ASAP prompt ID
        y_min: Minimum score
        y_max: Maximum score
        adapter_path: Optional path to LoRA adapter
        device: Device to use
        dtype: Data type

    Returns:
        Loaded ZeroShotScorer
    """
    scorer = ZeroShotScorer(
        model_name=model_name,
        y_min=y_min,
        y_max=y_max,
        prompt_id=prompt_id,
        device=device,
        dtype=dtype,
    )
    scorer.load_model(adapter_path)
    return scorer
