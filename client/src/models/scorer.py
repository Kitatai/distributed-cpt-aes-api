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
from transformers.cache_utils import DynamicCache
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


class ScoreTokenSets:
    """
    Pre-computed token sets and masks for score generation constraints.

    This class is created once per tokenizer and cached for reuse,
    avoiding repeated vocabulary scans.

    Token categories:
    - pure_digit_tokens: Tokens with ONLY digits, no spaces (e.g., "1", "12")
    - space_digit_tokens: Tokens with leading space + digits (e.g., " 1", " 12")
    - space_tokens: Pure whitespace tokens (e.g., " ", "  ")
    - newline_tokens: Tokens containing newline (e.g., "\\n")
    - boundary_tokens: Empty string tokens (SentencePiece markers)

    Rules:
    - First token: space | space_digit | pure_digit | boundary
    - After first (no digits yet): pure_digit only
    - After digits: pure_digit | newline

    Pre-computed masks are stored for fast GPU-based masking.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Build token sets and masks from tokenizer vocabulary."""
        self.space_tokens: set = set()
        self.pure_digit_tokens: set = set()
        self.space_digit_tokens: set = set()
        self.newline_tokens: set = set()
        self.boundary_tokens: set = set()
        self.digit_containing_tokens: set = set()  # Any token containing digits

        vocab_size = tokenizer.vocab_size
        for token_id in range(vocab_size):
            try:
                decoded = tokenizer.decode([token_id])

                # Boundary tokens: SentencePiece word boundary markers
                if decoded == "":
                    self.boundary_tokens.add(token_id)
                    continue

                # Space tokens: whitespace but NOT newline
                if decoded.strip() == "" and "\n" not in decoded:
                    self.space_tokens.add(token_id)
                    continue

                # Newline tokens: tokens containing newline
                if "\n" in decoded:
                    self.newline_tokens.add(token_id)
                    continue

                # Check if it's a digit token
                stripped = decoded.lstrip()
                if stripped and all(c.isdigit() for c in stripped):
                    # Track all digit-containing tokens for quick lookup
                    self.digit_containing_tokens.add(token_id)
                    # Distinguish between pure digits and space+digits
                    if decoded[0].isdigit():
                        # No leading space: pure digit token
                        self.pure_digit_tokens.add(token_id)
                    else:
                        # Has leading space: space+digit token
                        self.space_digit_tokens.add(token_id)

            except Exception:
                continue

        # Pre-compute combined sets for efficiency
        # First token: can start with space, space+digit, pure digit, or boundary
        self.valid_first_tokens = (
            self.space_tokens |
            self.space_digit_tokens |
            self.pure_digit_tokens |
            self.boundary_tokens
        )
        # After digits: only pure digits or newline (no more spaces allowed)
        self.valid_continuation = self.pure_digit_tokens | self.newline_tokens

        # Pre-compute boolean masks as tensors for fast GPU masking
        self._vocab_size = vocab_size
        self._masks_device = None
        self._first_token_mask = None
        self._continuation_mask = None
        self._pure_digit_mask = None

        logger.info(
            f"ScoreTokenSets built: pure_digit={len(self.pure_digit_tokens)}, "
            f"space_digit={len(self.space_digit_tokens)}, space={len(self.space_tokens)}, "
            f"newline={len(self.newline_tokens)}, boundary={len(self.boundary_tokens)}"
        )

    def _build_mask(self, token_set: set, device: torch.device) -> torch.Tensor:
        """Build a boolean mask tensor from a token set."""
        mask = torch.zeros(self._vocab_size, dtype=torch.bool, device=device)
        if token_set:
            indices = torch.tensor(list(token_set), dtype=torch.long, device=device)
            mask[indices] = True
        return mask

    def get_masks(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get pre-computed masks on the specified device.

        Returns:
            Tuple of (first_token_mask, continuation_mask, pure_digit_mask)
        """
        if self._masks_device != device:
            # Build masks on the correct device
            self._first_token_mask = self._build_mask(self.valid_first_tokens, device)
            self._continuation_mask = self._build_mask(self.valid_continuation, device)
            self._pure_digit_mask = self._build_mask(self.pure_digit_tokens, device)
            self._masks_device = device
        return self._first_token_mask, self._continuation_mask, self._pure_digit_mask

    def token_contains_digit(self, token_id: int) -> bool:
        """Check if a token contains any digit."""
        return token_id in self.digit_containing_tokens


class ConstrainedScoreLogitsProcessor(LogitsProcessor):
    """
    Logits processor that constrains output to valid score tokens only.

    Strict rules:
    - First token: space | space+digit | pure_digit | boundary
    - After first token (no digits yet): pure_digit only
    - After any digit appears: pure_digit | newline only

    This ensures output is exactly: [optional space][digits][newline]
    No spaces are allowed after the first token.

    Optimized for performance:
    - Uses pre-computed GPU tensors for masking (no Python loops)
    - Tracks digit state via token IDs instead of decoding (no tokenizer calls)

    Supports batched inference with per-sequence state tracking.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        input_length: Union[int, List[int]],
        token_sets: ScoreTokenSets,
    ):
        """
        Initialize the processor.

        Args:
            tokenizer: The tokenizer (unused, kept for API compatibility)
            input_length: Length of input tokens (single int or list for batch)
            token_sets: Pre-computed token sets (cached for efficiency)
        """
        # Support both single value and list for batched inference
        if isinstance(input_length, int):
            self.input_lengths = [input_length]
            self.batch_size = 1
        else:
            self.input_lengths = input_length
            self.batch_size = len(input_length)

        self.token_sets = token_sets
        # Track state per batch item
        self._has_seen_digit = [False] * self.batch_size
        self._masks_initialized = False
        self._first_mask = None
        self._cont_mask = None
        self._digit_mask = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to mask invalid tokens.

        Args:
            input_ids: Current sequence of token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]

        Returns:
            Modified logits with invalid tokens masked to -inf
        """
        batch_size = input_ids.shape[0]

        # Initialize masks on first call (on correct device)
        if not self._masks_initialized:
            self._first_mask, self._cont_mask, self._digit_mask = \
                self.token_sets.get_masks(scores.device)
            self._masks_initialized = True

        # Handle vocab size mismatch
        vocab_size = scores.shape[-1]
        first_mask = self._first_mask
        cont_mask = self._cont_mask
        digit_mask = self._digit_mask

        if vocab_size > first_mask.shape[-1]:
            # Extend masks with False for extra tokens
            first_mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            first_mask[:self._first_mask.shape[-1]] = self._first_mask
            cont_mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            cont_mask[:self._cont_mask.shape[-1]] = self._cont_mask
            digit_mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            digit_mask[:self._digit_mask.shape[-1]] = self._digit_mask

        # Process each batch item
        for i in range(batch_size):
            # Calculate how many tokens have been generated for this item
            generated_length = input_ids.shape[1] - self.input_lengths[i]

            if generated_length == 0:
                # First generated token: allow space, space+digit, pure digit, boundary
                mask = first_mask
            else:
                # Check if last generated token contained a digit (fast O(1) lookup)
                if not self._has_seen_digit[i]:
                    last_token = input_ids[i, -1].item()
                    if self.token_sets.token_contains_digit(last_token):
                        self._has_seen_digit[i] = True

                if self._has_seen_digit[i]:
                    # After digits: only pure digits or newline (no spaces)
                    mask = cont_mask
                else:
                    # After space/boundary but no digits yet: only pure digits
                    mask = digit_mask

            # Apply mask to this batch item
            scores[i] = scores[i].masked_fill(~mask, float('-inf'))

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
        dataset: str = "asap",
    ):
        """
        Initialize the scorer.

        Args:
            model_name: HuggingFace model name or path
            y_min: Minimum allowed score
            y_max: Maximum allowed score
            prompt_id: Prompt ID
            device: Device to use
            dtype: Data type for model
            use_generic_rubric: Whether to use generic rubric
            output_prefix: Output prefix for score
            dataset: Dataset type ("asap" or "toefl11")
        """
        self.model_name = model_name
        self.y_min = y_min
        self.y_max = y_max
        self.prompt_id = prompt_id
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.output_prefix = output_prefix
        self.dataset = dataset

        # Initialize components
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.prompt_builder: Optional[ScoringPromptBuilder] = None
        self.stopping_criteria: Optional[StoppingCriteriaList] = None
        self._adapter_loaded = False
        self._score_token_sets: Optional[ScoreTokenSets] = None

        # Create prompt builder
        self.prompt_builder = create_prompt_builder(
            prompt_id=prompt_id,
            y_min=y_min,
            y_max=y_max,
            use_generic_rubric=use_generic_rubric,
            output_prefix=output_prefix,
            dataset=dataset,
        )

        # Prefix caching state
        self._prefix_kv_cache = None
        self._prefix_length = None
        self._prefix_text = None
        self._prefix_cache_valid = False
        self._working_cache = None

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

        # Build token sets for constrained decoding (cached for efficiency)
        self._score_token_sets = ScoreTokenSets(self.tokenizer)

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

        # Invalidate prefix cache when adapter changes
        self._invalidate_prefix_cache()

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
            # Invalidate prefix cache when adapter changes
            self._invalidate_prefix_cache()
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
        self._adapter_loaded = hasattr(model, 'peft_config')

        # Invalidate prefix cache when model changes
        self._invalidate_prefix_cache()

        # Only rebuild token sets if tokenizer changed
        if self.tokenizer is not tokenizer:
            self.tokenizer = tokenizer
            # Setup stopping criteria
            self.stopping_criteria = StoppingCriteriaList([
                NewlineStoppingCriteria(self.tokenizer)
            ])
            # Build token sets for constrained decoding (cached for efficiency)
            self._score_token_sets = ScoreTokenSets(self.tokenizer)

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

    # =========================================================================
    # Prefix Caching Methods
    # =========================================================================

    def _invalidate_prefix_cache(self):
        """Invalidate the cached prefix KV cache."""
        self._prefix_kv_cache = None
        self._prefix_length = None
        self._prefix_text = None
        self._prefix_cache_valid = False
        self._working_cache = None  # Also invalidate working cache

    @torch.inference_mode()
    def compute_prefix_cache(self):
        """
        Compute and cache KV values for the common prefix.

        The prefix includes everything before the essay content:
        - System message
        - User message up to "[Essay]\n"

        This cache is reused for all essays with the same prompt_id,
        avoiding redundant computation of ~1000-1500 tokens per essay.

        Call this once before scoring multiple essays. The cache is
        automatically invalidated when the adapter changes.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build prefix messages (system + partial user message up to [Essay]\n)
        prefix_messages = self.prompt_builder.to_prefix_messages()

        # Apply chat template to prefix
        # Note: continue_final_message=True keeps the user message open
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )

        # Tokenize prefix
        prefix_inputs = self.tokenizer(
            prefix_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        prefix_inputs = {k: v.to(self.device) for k, v in prefix_inputs.items()}

        # Use DynamicCache for the forward pass
        prefix_cache = DynamicCache()
        outputs = self.model(
            **prefix_inputs,
            past_key_values=prefix_cache,
            use_cache=True,
        )

        # Store cache as DynamicCache (required by new transformers)
        self._prefix_kv_cache = outputs.past_key_values
        self._prefix_length = prefix_inputs['input_ids'].shape[1]
        self._prefix_text = prefix_text
        self._prefix_input_ids = prefix_inputs['input_ids']
        self._prefix_cache_valid = True

        logger.info(f"Prefix cache computed: {self._prefix_length} tokens")

    def _prepare_suffix_input(self, essay_text: str) -> Dict[str, torch.Tensor]:
        """
        Prepare suffix input for an essay (to be used with cached prefix).

        The suffix includes:
        - Essay text
        - Rest of user message ([Output Format]...)
        - Assistant prefill

        Args:
            essay_text: Essay text to score

        Returns:
            Dictionary with suffix input tensors
        """
        if not self._prefix_cache_valid:
            raise RuntimeError("Prefix cache not computed. Call compute_prefix_cache() first.")

        # Build full prompt to extract the suffix portion
        full_messages = self.prompt_builder.to_messages(essay_text, use_prefill=True)
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )

        # Extract suffix (everything after the prefix)
        suffix_text = full_text[len(self._prefix_text):]

        # Tokenize suffix without adding BOS token
        suffix_inputs = self.tokenizer(
            suffix_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            add_special_tokens=False,  # Don't add BOS again
        )

        return {k: v.to(self.device) for k, v in suffix_inputs.items()}

    def _is_score_determined(self, current_digits: str) -> bool:
        """
        Check if the score is already determined (no valid continuation possible).

        Early termination optimization: if appending any digit (0-9) would result
        in a value outside the valid score range, the current digits already
        represent the final score.

        For example, with range [2, 12]:
        - "5" → 50+ is out of range → score is 5, DONE
        - "1" → 10, 11, 12 are valid → need to continue

        Args:
            current_digits: String of digits accumulated so far

        Returns:
            True if score is determined (can terminate early)
        """
        if not current_digits:
            return False

        current_value = int(current_digits)

        # If current value is already out of range, something is wrong
        # but we should continue to get whatever output we can
        if current_value > self.y_max:
            return True

        # Check if any continuation is possible
        # min value with one more digit = current * 10 + 0
        min_with_continuation = current_value * 10

        # If even the minimum continuation exceeds max, score is determined
        return min_with_continuation > self.y_max

    @torch.inference_mode()
    def score_essay_with_prefix_cache(
        self,
        essay_text: str,
        max_new_tokens: int = 4,
    ) -> Tuple[Optional[int], bool, str]:
        """
        Score an essay using cached prefix KV values with manual greedy decoding.

        This is faster than score_essay_greedy when scoring multiple essays
        with the same prompt_id, as it avoids recomputing the common prefix.

        Uses manual greedy decoding loop instead of generate() for better
        compatibility with the new transformers Cache API.

        Includes early termination optimization: stops as soon as the score
        is determined (when no valid digit continuation is possible).

        Args:
            essay_text: Essay text to score
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Tuple of (score, parsed_ok, raw_text)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not self._prefix_cache_valid:
            raise RuntimeError("Prefix cache not computed. Call compute_prefix_cache() first.")

        # Prepare suffix input
        suffix_inputs = self._prepare_suffix_input(essay_text)

        # Use the working cache (created once, reused for all essays)
        # After each essay, we crop it back to prefix length
        if not hasattr(self, '_working_cache') or self._working_cache is None:
            # First call: clone the prefix cache using legacy format conversion
            legacy_cache = self._prefix_kv_cache.to_legacy_cache()
            # Deep clone the tensors
            cloned_legacy = tuple(
                tuple(t.clone() for t in layer)
                for layer in legacy_cache
            )
            self._working_cache = DynamicCache.from_legacy_cache(cloned_legacy)

        # Forward pass through suffix with working cache
        suffix_outputs = self.model(
            input_ids=suffix_inputs['input_ids'],
            past_key_values=self._working_cache,
            use_cache=True,
        )

        # Get logits for greedy decoding
        next_token_logits = suffix_outputs.logits[0, -1, :]

        # Get token masks for constrained decoding
        first_mask, cont_mask, digit_mask = self._score_token_sets.get_masks(self.device)

        # Manual greedy decoding with constraints and early termination
        generated_tokens = []
        has_seen_digit = False
        accumulated_digits = ""  # Track digits for early termination check

        for step in range(max_new_tokens):
            # Apply constraint mask
            if step == 0:
                # First token: space, space+digit, pure digit, or boundary
                mask = first_mask
            elif has_seen_digit:
                # After digits: only pure digits or newline
                mask = cont_mask
            else:
                # After space/boundary but no digits yet: only pure digits
                mask = digit_mask

            # Handle vocab size mismatch
            if next_token_logits.shape[0] > mask.shape[0]:
                extended_mask = torch.zeros(next_token_logits.shape[0], dtype=torch.bool, device=self.device)
                extended_mask[:mask.shape[0]] = mask
                mask = extended_mask

            # Apply mask
            masked_logits = next_token_logits.masked_fill(~mask, float('-inf'))

            # Greedy: take argmax
            next_token_id = torch.argmax(masked_logits).item()
            generated_tokens.append(next_token_id)

            # Update digit tracking and accumulate digits
            if self._score_token_sets.token_contains_digit(next_token_id):
                has_seen_digit = True
                # Extract digits from token
                token_text = self.tokenizer.decode([next_token_id])
                for c in token_text:
                    if c.isdigit():
                        accumulated_digits += c

            # Check for newline (stop condition)
            if next_token_id in self._score_token_sets.newline_tokens:
                break

            # Early termination: check if score is already determined
            if has_seen_digit and self._is_score_determined(accumulated_digits):
                # Score is determined, no need to generate newline
                break

            # Forward pass for next token
            next_token = torch.tensor([[next_token_id]], device=self.device)
            step_outputs = self.model(
                input_ids=next_token,
                past_key_values=self._working_cache,
                use_cache=True,
            )
            next_token_logits = step_outputs.logits[0, -1, :]

        # Crop the working cache back to prefix length for next essay
        self._working_cache.crop(self._prefix_length)

        # Decode generated tokens
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Parse score
        score, parsed_ok = self._parse_score(raw_text)

        return score, parsed_ok, raw_text

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

        # Create constrained logits processor (using cached token sets)
        logits_processor = LogitsProcessorList([
            ConstrainedScoreLogitsProcessor(
                self.tokenizer,
                input_length,
                self._score_token_sets,
            )
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

    def score_essays_batch(
        self,
        essay_texts: List[str],
        max_new_tokens: int = 4,
    ) -> List[Tuple[Optional[int], bool, str]]:
        """
        Score multiple essays in a single batch using constrained greedy decoding.

        Args:
            essay_texts: List of essay texts to score
            max_new_tokens: Maximum new tokens to generate

        Returns:
            List of tuples (score, parsed_ok, raw_text) for each essay
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not essay_texts:
            return []

        batch_size = len(essay_texts)

        # Prepare all inputs
        all_inputs = []
        input_lengths = []

        for essay_text in essay_texts:
            inputs = self._prepare_input(essay_text)
            all_inputs.append(inputs)
            input_lengths.append(inputs['input_ids'].shape[1])

        # Pad to same length (left padding for decoder-only models)
        max_length = max(input_lengths)
        padded_input_ids = []
        padded_attention_mask = []

        for inputs, orig_len in zip(all_inputs, input_lengths):
            pad_len = max_length - orig_len
            if pad_len > 0:
                # Left pad
                pad_ids = torch.full(
                    (1, pad_len),
                    self.tokenizer.pad_token_id,
                    dtype=inputs['input_ids'].dtype,
                    device=inputs['input_ids'].device,
                )
                pad_mask = torch.zeros(
                    (1, pad_len),
                    dtype=inputs['attention_mask'].dtype,
                    device=inputs['attention_mask'].device,
                )
                padded_input_ids.append(torch.cat([pad_ids, inputs['input_ids']], dim=1))
                padded_attention_mask.append(torch.cat([pad_mask, inputs['attention_mask']], dim=1))
            else:
                padded_input_ids.append(inputs['input_ids'])
                padded_attention_mask.append(inputs['attention_mask'])

        # Stack into batch tensors
        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_attention_mask = torch.cat(padded_attention_mask, dim=0)

        # Adjust input lengths for padding (since we left-padded)
        adjusted_input_lengths = [max_length] * batch_size

        # Create constrained logits processor (batch-aware)
        logits_processor = LogitsProcessorList([
            ConstrainedScoreLogitsProcessor(
                self.tokenizer,
                adjusted_input_lengths,
                self._score_token_sets,
            )
        ])

        # Generate with constrained decoding
        outputs = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=logits_processor,
            use_cache=True,
        )

        # Decode and parse results for each essay
        results = []
        for i in range(batch_size):
            # Get generated tokens (after the padded input)
            generated_tokens = outputs[i, max_length:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Parse score
            score, parsed_ok = self._parse_score(raw_text)
            results.append((score, parsed_ok, raw_text))

        return results

    def score_essay(
        self,
        essay_id: int,
        essay_text: str,
        y_true: Optional[int] = None,
        max_new_tokens: int = 12,
        compute_logit_expected: bool = False,
        logit_extractor=None,
        use_prefix_cache: bool = False,
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
            use_prefix_cache: Whether to use prefix caching (requires compute_prefix_cache() first)

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
            # Note: prefix caching not supported with unified extraction
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
            # Greedy decoding (with or without prefix cache)
            if use_prefix_cache and self._prefix_cache_valid:
                y_hat_greedy, parsed_ok, raw_text = self.score_essay_with_prefix_cache(
                    essay_text, max_new_tokens
                )
            else:
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
        use_prefix_cache: bool = True,  # Enabled by default for ~2.7x speedup
    ) -> List[ScoringResult]:
        """
        Score multiple essays.

        Automatically uses prefix caching for faster inference when scoring
        multiple essays with the same prompt_id. The common prefix (system
        message, rubric, etc.) is computed once and reused for all essays.

        Args:
            essays: List of dicts with 'essay_id', 'essay_text', and optionally 'score'
            max_new_tokens: Maximum tokens for greedy decoding
            compute_logit_expected: Whether to compute logit-based expected value
            logit_extractor: LogitExpectedValueExtractor instance
            show_progress: Whether to show progress bar
            use_prefix_cache: Whether to use prefix caching (default: True)

        Returns:
            List of ScoringResult objects
        """
        # Compute prefix cache if enabled and not using unified extraction
        use_unified = (
            compute_logit_expected and
            logit_extractor is not None and
            hasattr(logit_extractor, 'extract_unified')
        )

        if use_prefix_cache and not use_unified and len(essays) > 0:
            # Compute prefix cache once for all essays
            self.compute_prefix_cache()

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
                use_prefix_cache=use_prefix_cache and not use_unified,
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
