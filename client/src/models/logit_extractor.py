"""
Logit-based expected value extractor for essay scoring.

Implements unified path enumeration with probability-based pruning
to compute probability distribution over scores and expected value.
Works consistently across all tokenizer types (Llama, Mistral, Qwen, etc.).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TokenPath:
    """Container for a single token path."""
    token_ids: List[int]
    decoded: str
    log_prob: float
    mapped_score: Optional[int]
    terminated_by_newline: bool


@dataclass
class LogitExtractionResult:
    """Container for logit extraction results."""
    y_hat_argmax: Optional[int]
    y_tilde: Optional[float]
    y_tilde_round: Optional[int]
    p_valid_sum: float
    p_invalid: float
    logp_by_score: Dict[str, float]
    paths: Optional[List[TokenPath]] = None
    # Greedy result (for unified extraction)
    y_hat_greedy: Optional[int] = None
    greedy_token_id: Optional[int] = None
    greedy_decoded: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            'y_hat_greedy': self.y_hat_greedy,
            'y_hat_argmax': self.y_hat_argmax,
            'y_tilde': self.y_tilde,
            'y_tilde_round': self.y_tilde_round,
            'p_valid_sum': self.p_valid_sum,
            'p_invalid': self.p_invalid,
            'logp_by_score': self.logp_by_score,
        }
        if self.paths:
            result['paths'] = [
                {
                    'token_ids': p.token_ids,
                    'decoded': p.decoded,
                    'log_prob': p.log_prob,
                    'mapped_score': p.mapped_score,
                    'terminated_by_newline': p.terminated_by_newline,
                }
                for p in self.paths
            ]
        return result


class UnifiedLogitExtractor:
    """
    Unified logit extractor for score probability distribution.

    Key design principles:
    1. No model-specific branching - works with any tokenizer
    2. Probability-based pruning for efficiency
    3. Handles all token patterns uniformly:
       - Direct digit tokens (e.g., "5", "10")
       - Space + digit sequences (e.g., " ", "5")
       - Multi-token digits (e.g., "1", "0" for "10")
    4. Accumulates all paths leading to the same score

    Algorithm:
    - BFS over token sequences
    - At each step, consider valid continuations (space, digits, newline)
    - Prune paths when cumulative probability < threshold
    - Parse decoded strings to map paths to scores
    - Sum probabilities for paths leading to the same score
    """

    def __init__(
        self,
        model,
        tokenizer,
        y_min: int,
        y_max: int,
        max_steps: int = 4,
        prob_threshold: float = 1e-6,
        device: str = "cuda",
    ):
        """
        Initialize the extractor.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            y_min: Minimum allowed score
            y_max: Maximum allowed score
            max_steps: Maximum depth for path enumeration
            prob_threshold: Minimum cumulative probability to continue exploration
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.y_min = y_min
        self.y_max = y_max
        self.max_steps = max_steps
        self.prob_threshold = prob_threshold
        self.device = device

        # Pre-compute valid token sets
        self._space_tokens: Set[int] = set()
        self._digit_tokens: Set[int] = set()  # Tokens containing only digits
        self._newline_tokens: Set[int] = set()
        self._boundary_tokens: Set[int] = set()  # SentencePiece word boundary markers
        self._build_token_sets()

        logger.info(
            f"UnifiedLogitExtractor initialized: "
            f"y_range=[{y_min}, {y_max}], "
            f"max_steps={max_steps}, "
            f"prob_threshold={prob_threshold}, "
            f"space_tokens={len(self._space_tokens)}, "
            f"digit_tokens={len(self._digit_tokens)}, "
            f"newline_tokens={len(self._newline_tokens)}, "
            f"boundary_tokens={len(self._boundary_tokens)}"
        )

    def _build_token_sets(self):
        """Pre-compute sets of valid tokens.

        Handles different tokenizer types uniformly:
        - Llama-style: digits have dedicated tokens, space is separate
        - SentencePiece-style (Mistral): has word boundary markers that decode to ''
        """
        vocab_size = self.tokenizer.vocab_size

        for token_id in range(vocab_size):
            try:
                decoded = self.tokenizer.decode([token_id])

                # Boundary tokens: SentencePiece word boundary markers
                # These decode to empty string and act as "start of word" markers
                if decoded == "":
                    self._boundary_tokens.add(token_id)
                    continue

                # Space tokens: tokens that are just whitespace (space, tab, etc.)
                # but NOT newline
                if decoded.strip() == "" and "\n" not in decoded:
                    self._space_tokens.add(token_id)

                # Digit tokens: tokens that contain only digits (possibly with leading space)
                stripped = decoded.lstrip()
                if stripped and all(c.isdigit() for c in stripped):
                    self._digit_tokens.add(token_id)

                # Newline tokens: tokens containing newline
                if "\n" in decoded:
                    self._newline_tokens.add(token_id)

            except Exception:
                continue

    def _get_valid_first_tokens(self) -> Set[int]:
        """Get tokens valid as first token.

        Includes:
        - Space tokens (for models that output space first due to chat template)
        - Digit tokens (direct digit output)
        - Boundary tokens (SentencePiece word boundary markers)
        """
        return self._space_tokens | self._digit_tokens | self._boundary_tokens

    def _get_valid_continuation_tokens(self, has_digits: bool) -> Set[int]:
        """
        Get tokens valid as continuation.

        Args:
            has_digits: Whether we already have digits in the path

        Returns:
            Set of valid token IDs
        """
        if has_digits:
            # After digits: more digits or newline (to terminate)
            return self._digit_tokens | self._newline_tokens
        else:
            # Before digits: only digits allowed (space/newline/boundary already handled)
            return self._digit_tokens

    def _decode_to_score(self, decoded: str) -> Optional[int]:
        """
        Parse decoded string to integer score.

        Args:
            decoded: Decoded token string

        Returns:
            Integer score if valid and in range, None otherwise
        """
        stripped = decoded.strip()
        if not stripped:
            return None

        # Extract leading digits (before newline or other chars)
        digits = ""
        for char in stripped:
            if char.isdigit():
                digits += char
            else:
                break

        if not digits:
            return None

        try:
            score = int(digits)
            if self.y_min <= score <= self.y_max:
                return score
            return None
        except ValueError:
            return None

    def _has_digits(self, decoded: str) -> bool:
        """Check if decoded string contains any digits."""
        return any(c.isdigit() for c in decoded)

    def _is_boundary_token(self, token_id: int) -> bool:
        """Check if token is a SentencePiece boundary marker."""
        return token_id in self._boundary_tokens

    def _can_extend_to_valid_score(self, decoded: str) -> bool:
        """
        Check if decoded string can be extended to another valid score.

        For score range [2, 12]:
        - "2"-"9" cannot extend to valid score (20, 30, etc. are out of range)
        - "1" can extend to 10, 11, 12
        - "10", "11", "12" cannot extend further

        This allows us to terminate early for confirmed scores.
        """
        # Extract digits from decoded string
        digits = ""
        for c in decoded.strip():
            if c.isdigit():
                digits += c
            else:
                break

        if not digits:
            return True  # No digits yet, can extend

        # Check if any single-digit extension would be valid
        for next_digit in "0123456789":
            extended = digits + next_digit
            try:
                extended_score = int(extended)
                if self.y_min <= extended_score <= self.y_max:
                    return True
            except ValueError:
                pass

        return False

    @torch.no_grad()
    def extract_unified(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        save_paths: bool = False,
    ) -> LogitExtractionResult:
        """
        Extract score distribution AND greedy prediction using unified algorithm.

        Args:
            input_ids: Input token IDs (with prompt and prefill)
            attention_mask: Attention mask
            save_paths: Whether to save individual paths for debugging

        Returns:
            LogitExtractionResult with both greedy and expected value
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Dictionary to accumulate probabilities by score
        score_probs: Dict[int, float] = defaultdict(float)
        all_paths: List[TokenPath] = []

        # Track greedy path
        greedy_token_ids: List[int] = []
        greedy_decoded: str = ""

        # BFS queue: (current_input_ids, current_attention_mask, decoded_str,
        #             cumulative_prob, token_ids_path, is_greedy_path, has_boundary)
        initial_state = (
            input_ids.clone(),
            attention_mask.clone() if attention_mask is not None else None,
            "",  # decoded string
            1.0,  # cumulative probability
            [],  # token IDs in path
            True,  # is this the greedy path?
            False,  # has seen boundary token
        )
        queue = [initial_state]

        for step in range(self.max_steps):
            next_queue = []

            for curr_ids, curr_mask, curr_decoded, curr_prob, path_ids, is_greedy, has_boundary in queue:
                # Prune low probability paths
                if curr_prob < self.prob_threshold:
                    continue

                # Get next token distribution
                outputs = self.model(input_ids=curr_ids, attention_mask=curr_mask)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

                # Get greedy token for this position
                greedy_token_id = logits.argmax().item()

                # Determine valid continuations based on current state
                has_digits = self._has_digits(curr_decoded)
                if step == 0:
                    valid_tokens = self._get_valid_first_tokens()
                else:
                    valid_tokens = self._get_valid_continuation_tokens(has_digits)

                # Explore valid continuations
                for token_id in valid_tokens:
                    token_prob = probs[token_id].item()
                    new_prob = curr_prob * token_prob

                    # Skip very low probability tokens
                    if new_prob < self.prob_threshold:
                        continue

                    # Decode token
                    token_decoded = self.tokenizer.decode([token_id])
                    new_decoded = curr_decoded + token_decoded
                    new_path_ids = path_ids + [token_id]

                    # Check if this is (part of) the greedy path
                    new_is_greedy = is_greedy and (token_id == greedy_token_id)

                    # Update greedy tracking
                    if new_is_greedy:
                        greedy_token_ids = new_path_ids.copy()
                        greedy_decoded = new_decoded

                    # Check termination
                    is_newline = token_id in self._newline_tokens
                    score = self._decode_to_score(new_decoded)

                    # Terminate if:
                    # 1. Newline found (explicit termination), OR
                    # 2. Valid score that cannot be extended to another valid score
                    #    e.g., "5" terminates (50 > y_max), "1" continues (10, 11, 12 valid)
                    can_extend = self._can_extend_to_valid_score(new_decoded)
                    should_terminate = is_newline or (score is not None and not can_extend)

                    if should_terminate:
                        # Record path
                        if save_paths:
                            path = TokenPath(
                                token_ids=new_path_ids,
                                decoded=new_decoded,
                                log_prob=np.log(new_prob) if new_prob > 0 else float('-inf'),
                                mapped_score=score,
                                terminated_by_newline=is_newline,
                            )
                            all_paths.append(path)

                        # Accumulate probability for this score
                        if score is not None:
                            score_probs[score] += new_prob
                    else:
                        # Continue exploration
                        new_ids = torch.cat([
                            curr_ids,
                            torch.tensor([[token_id]], device=self.device)
                        ], dim=1)

                        if curr_mask is not None:
                            new_mask = torch.cat([
                                curr_mask,
                                torch.ones((1, 1), device=self.device, dtype=curr_mask.dtype)
                            ], dim=1)
                        else:
                            new_mask = None

                        # Update boundary flag if this token is a boundary marker
                        new_has_boundary = has_boundary or self._is_boundary_token(token_id)

                        next_queue.append((
                            new_ids, new_mask, new_decoded, new_prob,
                            new_path_ids, new_is_greedy, new_has_boundary
                        ))

            queue = next_queue

            if not queue:
                break

        # Handle remaining paths at max depth
        for curr_ids, curr_mask, curr_decoded, curr_prob, path_ids, is_greedy, _ in queue:
            score = self._decode_to_score(curr_decoded)

            if save_paths:
                path = TokenPath(
                    token_ids=path_ids,
                    decoded=curr_decoded,
                    log_prob=np.log(curr_prob) if curr_prob > 0 else float('-inf'),
                    mapped_score=score,
                    terminated_by_newline=False,
                )
                all_paths.append(path)

            if score is not None:
                score_probs[score] += curr_prob

        # Compute log probabilities
        logp_by_score: Dict[str, float] = {}
        for score, prob in score_probs.items():
            if prob > 0:
                logp_by_score[str(score)] = np.log(prob)

        # Compute statistics
        p_valid_sum = sum(score_probs.values())
        p_invalid = 1.0 - p_valid_sum

        if p_valid_sum > 0:
            # Normalize and compute expected value
            p_normalized = {s: p / p_valid_sum for s, p in score_probs.items()}
            y_tilde = sum(s * p for s, p in p_normalized.items())
            y_tilde_round = round(y_tilde)
            y_hat_argmax = max(score_probs.keys(), key=lambda s: score_probs[s])
        else:
            y_tilde = None
            y_tilde_round = None
            y_hat_argmax = None

        # Parse greedy score
        y_hat_greedy = self._decode_to_score(greedy_decoded)

        # Fallback: use argmax if greedy parsing failed
        if y_hat_greedy is None and y_hat_argmax is not None:
            y_hat_greedy = y_hat_argmax

        return LogitExtractionResult(
            y_hat_argmax=y_hat_argmax,
            y_tilde=y_tilde,
            y_tilde_round=y_tilde_round,
            p_valid_sum=p_valid_sum,
            p_invalid=p_invalid,
            logp_by_score=logp_by_score,
            paths=all_paths if save_paths else None,
            y_hat_greedy=y_hat_greedy,
            greedy_token_id=greedy_token_ids[0] if greedy_token_ids else None,
            greedy_decoded=greedy_decoded,
        )

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        save_paths: bool = False,
    ) -> LogitExtractionResult:
        """
        Extract score distribution (alias for extract_unified).

        Args:
            input_ids: Input token IDs (with prompt and prefill)
            attention_mask: Attention mask
            save_paths: Whether to save individual paths

        Returns:
            LogitExtractionResult
        """
        return self.extract_unified(input_ids, attention_mask, save_paths)


def create_logit_extractor(
    model,
    tokenizer,
    y_min: int,
    y_max: int,
    max_steps: int = 4,
    prob_threshold: float = 1e-6,
    device: str = "cuda",
) -> UnifiedLogitExtractor:
    """
    Create logit extractor.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        y_min: Minimum score
        y_max: Maximum score
        max_steps: Maximum search depth
        prob_threshold: Minimum probability threshold for path exploration
        device: Device

    Returns:
        UnifiedLogitExtractor instance
    """
    return UnifiedLogitExtractor(
        model=model,
        tokenizer=tokenizer,
        y_min=y_min,
        y_max=y_max,
        max_steps=max_steps,
        prob_threshold=prob_threshold,
        device=device,
    )
