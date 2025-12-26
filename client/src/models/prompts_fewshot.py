"""
Few-shot scoring prompts module for essay scoring.

Extends the zero-shot prompt builder to include example essays with scores.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .prompts import (
    ScoringPrompt,
    ScoringPromptBuilder,
    ASAP_PROMPT_TEXTS,
    ASAP_RUBRICS,
    GENERIC_HOLISTIC_RUBRIC,
)


@dataclass
class FewShotExample:
    """Container for a few-shot example."""
    essay_text: str
    score: int
    essay_id: Optional[int] = None


class FewShotScoringPromptBuilder(ScoringPromptBuilder):
    """
    Few-shot essay scoring prompt builder.

    Extends ScoringPromptBuilder to include example essays with their scores
    before the target essay.
    """

    def __init__(
        self,
        y_min: int,
        y_max: int,
        prompt_text: Optional[str] = None,
        rubric_text: Optional[str] = None,
        output_prefix: str = "The score of this essay:",
        include_reasoning: bool = False,
        examples: Optional[List[FewShotExample]] = None,
    ):
        """
        Initialize the few-shot prompt builder.

        Args:
            y_min: Minimum allowed score
            y_max: Maximum allowed score
            prompt_text: Writing prompt text (task description for students)
            rubric_text: Scoring rubric text
            output_prefix: Prefix for assistant output
            include_reasoning: Whether to include reasoning in output
            examples: List of FewShotExample objects (essays with scores)
        """
        super().__init__(
            y_min=y_min,
            y_max=y_max,
            prompt_text=prompt_text,
            rubric_text=rubric_text,
            output_prefix=output_prefix,
            include_reasoning=include_reasoning,
        )
        self.examples = examples or []

    def set_examples(self, examples: List[FewShotExample]):
        """
        Set few-shot examples.

        Args:
            examples: List of FewShotExample objects
        """
        self.examples = examples

    def add_example(self, essay_text: str, score: int, essay_id: Optional[int] = None):
        """
        Add a single few-shot example.

        Args:
            essay_text: Essay text
            score: Score for the essay
            essay_id: Optional essay ID
        """
        self.examples.append(FewShotExample(
            essay_text=essay_text,
            score=score,
            essay_id=essay_id,
        ))

    def clear_examples(self):
        """Clear all examples."""
        self.examples = []

    def _build_examples_section(self) -> str:
        """Build the examples section of the prompt."""
        if not self.examples:
            return ""

        lines = ["[Examples]"]
        lines.append("Here are some example essays with their scores:\n")

        for i, example in enumerate(self.examples, 1):
            lines.append(f"--- Example {i} ---")
            lines.append(f"Essay:\n{example.essay_text}")
            lines.append(f"\n{self.output_prefix} {example.score}")
            lines.append("")  # Empty line between examples

        return "\n".join(lines)

    def build_user_message(self, essay_text: str) -> str:
        """
        Build the user message with task, prompt, rubric, examples, and essay.

        Args:
            essay_text: The essay to be scored
        """
        examples_section = self._build_examples_section()

        # Build message with optional examples section
        if examples_section:
            message = f"""[Task]
You will score a student essay for the following writing prompt.
Study the examples carefully and score the target essay using the same criteria.

[Writing Prompt]
{self.prompt_text}

[Scoring Rubric]
{self.rubric_text}

[Allowed Score Range]
An integer score in [{self.y_min}, {self.y_max}] (inclusive).

{examples_section}

[Target Essay to Score]
{essay_text}

[Output Format]
{self.output_prefix}<INTEGER>"""
        else:
            # Fallback to zero-shot format if no examples
            message = super().build_user_message(essay_text)

        return message


def create_fewshot_prompt_builder(
    prompt_id: int,
    y_min: int,
    y_max: int,
    examples: Optional[List[FewShotExample]] = None,
    use_generic_rubric: bool = False,
    output_prefix: str = "The score of this essay:",
) -> FewShotScoringPromptBuilder:
    """
    Create a few-shot prompt builder for a specific ASAP prompt.

    Args:
        prompt_id: ASAP prompt ID (1-8)
        y_min: Minimum score
        y_max: Maximum score
        examples: List of FewShotExample objects
        use_generic_rubric: Whether to use generic rubric
        output_prefix: Output prefix for score

    Returns:
        FewShotScoringPromptBuilder configured for the prompt
    """
    prompt_text = ASAP_PROMPT_TEXTS.get(prompt_id, "")

    if use_generic_rubric or not ASAP_RUBRICS.get(prompt_id):
        rubric_text = GENERIC_HOLISTIC_RUBRIC
    else:
        rubric_text = ASAP_RUBRICS[prompt_id]

    return FewShotScoringPromptBuilder(
        y_min=y_min,
        y_max=y_max,
        prompt_text=prompt_text,
        rubric_text=rubric_text,
        output_prefix=output_prefix,
        examples=examples,
    )


def create_examples_from_essays(
    essays: List[Dict],
    n_examples: int = 3,
    strategy: str = "first",
) -> List[FewShotExample]:
    """
    Create few-shot examples from a list of essays.

    Args:
        essays: List of dicts with 'essay_text' (or 'essay'), 'score' (or 'domain1_score'),
                and optionally 'essay_id'
        n_examples: Number of examples to create
        strategy: Selection strategy
            - "first": Take first n essays
            - "random": Randomly select n essays
            - "diverse_scores": Select essays with diverse scores

    Returns:
        List of FewShotExample objects
    """
    import random

    # Normalize essay format
    normalized = []
    for e in essays:
        text = e.get('essay_text') or e.get('essay', '')
        score = e.get('score') or e.get('domain1_score', 0)
        essay_id = e.get('essay_id')
        normalized.append({
            'essay_text': text,
            'score': int(score),
            'essay_id': essay_id,
        })

    if not normalized:
        return []

    n_examples = min(n_examples, len(normalized))

    if strategy == "first":
        selected = normalized[:n_examples]
    elif strategy == "random":
        selected = random.sample(normalized, n_examples)
    elif strategy == "diverse_scores":
        # Sort by score and select evenly spaced examples
        sorted_essays = sorted(normalized, key=lambda x: x['score'])
        if len(sorted_essays) <= n_examples:
            selected = sorted_essays
        else:
            step = len(sorted_essays) / n_examples
            indices = [int(i * step) for i in range(n_examples)]
            selected = [sorted_essays[i] for i in indices]
    else:
        selected = normalized[:n_examples]

    return [
        FewShotExample(
            essay_text=e['essay_text'],
            score=e['score'],
            essay_id=e.get('essay_id'),
        )
        for e in selected
    ]
