"""
Scoring prompts module for zero-shot essay scoring.

Contains prompt templates and builders for LLM-based essay scoring.
Designed to be extensible with prompt-specific rubrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Base directory for prompt files
_PROMPTS_BASE_DIR = Path(__file__).parent.parent.parent / "exp" / "llm_prompts"


def _load_prompt_text(prompt_id: int, dataset: str = "asap") -> str:
    """Load essay prompt text from file."""
    if dataset == "toefl11":
        prompt_file = _PROMPTS_BASE_DIR / "toefl11_essay_prompts" / f"prompt_{prompt_id}.md"
    else:
        prompt_file = _PROMPTS_BASE_DIR / "essay_prompts" / f"prompt_{prompt_id}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return ""


def _load_rubric_text(prompt_id: int, dataset: str = "asap") -> str:
    """Load rubric text from file."""
    if dataset == "toefl11":
        # TOEFL11 uses a shared rubric for all prompts
        rubric_file = _PROMPTS_BASE_DIR / "toefl11_overall" / "Overall.md"
    else:
        rubric_file = _PROMPTS_BASE_DIR / "overall" / f"Overall_{prompt_id}.md"
    if rubric_file.exists():
        return rubric_file.read_text(encoding="utf-8").strip()
    return ""


def _load_all_prompts(dataset: str = "asap") -> Dict[int, str]:
    """Load all essay prompt texts from files."""
    prompts = {}
    for prompt_id in range(1, 9):
        text = _load_prompt_text(prompt_id, dataset)
        if text:
            prompts[prompt_id] = text
    return prompts


def _load_all_rubrics(dataset: str = "asap") -> Dict[int, str]:
    """Load all rubric texts from files."""
    rubrics = {}
    for prompt_id in range(1, 9):
        text = _load_rubric_text(prompt_id, dataset)
        if text:
            rubrics[prompt_id] = text
    return rubrics


# Load ASAP prompts and rubrics from files
ASAP_PROMPT_TEXTS: Dict[int, str] = _load_all_prompts("asap")
ASAP_RUBRICS: Dict[int, str] = _load_all_rubrics("asap")

# Load TOEFL11 prompts and rubrics from files
TOEFL11_PROMPT_TEXTS: Dict[int, str] = _load_all_prompts("toefl11")
TOEFL11_RUBRICS: Dict[int, str] = _load_all_rubrics("toefl11")


# Generic holistic rubric (used as fallback when prompt-specific rubric is not available)
GENERIC_HOLISTIC_RUBRIC = """Score the essay holistically considering:

* Content & relevance to the prompt
* Organization & coherence
* Evidence/support and clarity
* Grammar, vocabulary, and mechanics
* Overall effectiveness"""


@dataclass
class ScoringPrompt:
    """Container for a complete scoring prompt."""
    system_message: str
    user_message: str
    assistant_prefill: str


class ScoringPromptBuilder:
    """
    Builder for zero-shot essay scoring prompts.

    Creates prompts following the format specified in the requirements:
    - System: Role assignment and output constraints
    - User: Task description, writing prompt, rubric, score range, output format, essay
    - Assistant: Prefill with output prefix
    """

    def __init__(
        self,
        y_min: int,
        y_max: int,
        prompt_text: Optional[str] = None,
        rubric_text: Optional[str] = None,
        output_prefix: str = "The score of this essay:",
        include_reasoning: bool = False,
    ):
        """
        Initialize the prompt builder.

        Args:
            y_min: Minimum allowed score
            y_max: Maximum allowed score
            prompt_text: Writing prompt text (task description for students)
            rubric_text: Scoring rubric text
            output_prefix: Prefix for assistant output
            include_reasoning: Whether to include reasoning in output (not used in basic setting)
        """
        self.y_min = y_min
        self.y_max = y_max
        self.prompt_text = prompt_text or ""
        self.rubric_text = rubric_text or GENERIC_HOLISTIC_RUBRIC
        self.output_prefix = output_prefix
        self.include_reasoning = include_reasoning

    def build_system_message(self) -> str:
        """Build the system message with role and constraints."""
        return f"""You are a strict automated essay scoring engine.
Output ONLY the integer score, then a newline.
Do not output any other words, explanations, or punctuation.
The integer MUST be within [{self.y_min}, {self.y_max}]."""

    def build_user_message(self, essay_text: str) -> str:
        """
        Build the user message with task, prompt, rubric, and essay.

        Args:
            essay_text: The essay to be scored
        """
        message = f"""[Task]
You will score a student essay for the following writing prompt.

[Writing Prompt]
{self.prompt_text}

[Scoring Rubric]
{self.rubric_text}

[Allowed Score Range]
An integer score in [{self.y_min}, {self.y_max}] (inclusive).

[Essay]
{essay_text}

[Output Format]
{self.output_prefix}<INTEGER>"""
        return message

    def build_user_message_prefix(self) -> str:
        """
        Build the prefix part of user message (before essay).

        This is the cacheable part that remains constant for all essays
        with the same prompt_id.

        Returns:
            User message prefix ending with "[Essay]\n"
        """
        return f"""[Task]
You will score a student essay for the following writing prompt.

[Writing Prompt]
{self.prompt_text}

[Scoring Rubric]
{self.rubric_text}

[Allowed Score Range]
An integer score in [{self.y_min}, {self.y_max}] (inclusive).

[Essay]
"""

    def build_user_message_suffix(self, essay_text: str) -> str:
        """
        Build the suffix part of user message (essay and output format).

        This is the variable part that changes for each essay.

        Args:
            essay_text: The essay to be scored

        Returns:
            User message suffix with essay and output format
        """
        return f"""{essay_text}

[Output Format]
{self.output_prefix}<INTEGER>"""

    def build_assistant_prefill(self) -> str:
        """Build the assistant prefill (continuation point)."""
        return self.output_prefix

    def build(self, essay_text: str) -> ScoringPrompt:
        """
        Build the complete scoring prompt.

        Args:
            essay_text: The essay to be scored

        Returns:
            ScoringPrompt with system, user, and assistant prefill
        """
        return ScoringPrompt(
            system_message=self.build_system_message(),
            user_message=self.build_user_message(essay_text),
            assistant_prefill=self.build_assistant_prefill(),
        )

    def to_messages(self, essay_text: str, use_prefill: bool = True) -> List[Dict[str, str]]:
        """
        Convert to chat messages format.

        Args:
            essay_text: The essay to be scored
            use_prefill: Whether to include assistant prefill

        Returns:
            List of message dictionaries for chat API
        """
        prompt = self.build(essay_text)
        messages = [
            {"role": "system", "content": prompt.system_message},
            {"role": "user", "content": prompt.user_message},
        ]
        if use_prefill:
            messages.append({"role": "assistant", "content": prompt.assistant_prefill})
        return messages

    def to_prefix_messages(self) -> List[Dict[str, str]]:
        """
        Build messages for prefix caching (everything before essay content).

        The prefix includes system message and user message up to "[Essay]\n".
        This is constant for all essays with the same prompt_id.

        Returns:
            List of message dictionaries for the cacheable prefix
        """
        return [
            {"role": "system", "content": self.build_system_message()},
            {"role": "user", "content": self.build_user_message_prefix()},
        ]

    def to_suffix_text(self, essay_text: str) -> str:
        """
        Build the suffix text (essay content + output format + assistant prefill).

        This is the variable part that changes for each essay.
        Note: This returns raw text, not messages, because it continues
        the user message from the prefix.

        Args:
            essay_text: The essay to be scored

        Returns:
            Suffix text to append after the cached prefix
        """
        # Essay content + output format section + assistant turn with prefill
        return self.build_user_message_suffix(essay_text)


def create_prompt_builder(
    prompt_id: int,
    y_min: int,
    y_max: int,
    use_generic_rubric: bool = False,
    output_prefix: str = "The score of this essay:",
    dataset: str = "asap",
) -> ScoringPromptBuilder:
    """
    Create a prompt builder for a specific prompt.

    Args:
        prompt_id: Prompt ID (1-8)
        y_min: Minimum score
        y_max: Maximum score
        use_generic_rubric: Whether to use generic rubric (True) or prompt-specific (False)
        output_prefix: Output prefix for score
        dataset: Dataset type ("asap" or "toefl11")

    Returns:
        ScoringPromptBuilder configured for the prompt
    """
    if dataset == "toefl11":
        prompt_texts = TOEFL11_PROMPT_TEXTS
        rubrics = TOEFL11_RUBRICS
    else:
        prompt_texts = ASAP_PROMPT_TEXTS
        rubrics = ASAP_RUBRICS

    prompt_text = prompt_texts.get(prompt_id, "")

    if use_generic_rubric or not rubrics.get(prompt_id):
        rubric_text = GENERIC_HOLISTIC_RUBRIC
    else:
        rubric_text = rubrics[prompt_id]

    return ScoringPromptBuilder(
        y_min=y_min,
        y_max=y_max,
        prompt_text=prompt_text,
        rubric_text=rubric_text,
        output_prefix=output_prefix,
    )


def set_prompt_rubric(prompt_id: int, rubric_text: str):
    """
    Set a prompt-specific rubric.

    Args:
        prompt_id: ASAP prompt ID
        rubric_text: Rubric text to set
    """
    if prompt_id in ASAP_RUBRICS:
        ASAP_RUBRICS[prompt_id] = rubric_text


def get_prompt_info(prompt_id: int) -> Dict:
    """
    Get information about a specific ASAP prompt.

    Args:
        prompt_id: ASAP prompt ID

    Returns:
        Dictionary with prompt information
    """
    from ..config import ASAP_SCORE_RANGES
    score_range = ASAP_SCORE_RANGES.get(prompt_id, (0, 10))
    return {
        'prompt_id': prompt_id,
        'prompt_text': ASAP_PROMPT_TEXTS.get(prompt_id, ""),
        'rubric_text': ASAP_RUBRICS.get(prompt_id, ""),
        'has_rubric': bool(ASAP_RUBRICS.get(prompt_id)),
        'score_range': score_range,
    }
