"""Shared OPSD-style prompt formatting and math answer grading helpers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


LEGACY_DAPO_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    '(without quotes) where $Answer is the answer to the problem.\n\n'
    'Remember to put your answer on its own line after "Answer:".'
)

BOXED_ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Optional explicit reasoning instruction.
THINKING_BOXED_ANSWER_INSTRUCTION = (
    "Please first think step by step inside <think>...</think> tags, "
    "and then put your final answer within \\boxed{}."
)

TEACHER_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you understand "
    "the reasoning behind each step.\n"
)
EVAL_DATA_SOURCES = {"amc23", "aime24", "aime25", "hmmt25"}

_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DEGREE_SUFFIX_RE = re.compile(r"(?:\^\{?\\circ\}?|\\degree|\\deg|\u00b0)$")


def strip_legacy_math_prompt(text: str) -> str:
    """Strip known prompt wrappers and keep only the raw problem text."""
    if text is None:
        return ""

    text = str(text).strip()
    legacy_prefix = f"{LEGACY_DAPO_INSTRUCTION}\n\n"
    if text.startswith(legacy_prefix):
        return text[len(legacy_prefix):].strip()

    train_prefix = "Problem: "
    for instruction in (BOXED_ANSWER_INSTRUCTION, THINKING_BOXED_ANSWER_INSTRUCTION):
        suffix = f"\n\n{instruction}"
        if text.startswith(train_prefix) and text.endswith(suffix):
            return text[len(train_prefix):-len(suffix)].strip()
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()

    return text


def _answer_instruction(thinking: bool = False) -> str:
    return THINKING_BOXED_ANSWER_INSTRUCTION if thinking else BOXED_ANSWER_INSTRUCTION


def build_train_rollout_prompt(problem: str, thinking: bool = False) -> str:
    return f"Problem: {strip_legacy_math_prompt(problem)}\n\n{_answer_instruction(thinking)}"


def build_eval_rollout_prompt(problem: str, thinking: bool = False) -> str:
    return f"{strip_legacy_math_prompt(problem)}\n\n{_answer_instruction(thinking)}"


def build_rollout_prompt(problem: str, data_source: Optional[str] = None, thinking: bool = False) -> str:
    problem = strip_legacy_math_prompt(problem)
    if str(data_source or "").lower() in EVAL_DATA_SOURCES:
        return build_eval_rollout_prompt(problem, thinking=thinking)
    return build_train_rollout_prompt(problem, thinking=thinking)


def build_rollout_messages(problem: str, data_source: Optional[str] = None, thinking: Optional[bool] = None) -> list[dict]:
    # The native model chat template controls thinking.
    del thinking
    return [{"role": "user", "content": build_rollout_prompt(problem, data_source=data_source)}]


def normalize_privileged_text_mode(mode: str) -> str:
    normalized = str(mode or "solution_answer").strip().lower().replace("-", "_")
    if normalized in {"solution+answer", "solution_and_answer"}:
        normalized = "solution_answer"
    if normalized in {"answer", "answer_only"}:
        normalized = "answer_only"
    if normalized not in {"solution_answer", "answer_only"}:
        raise ValueError(
            f"Unsupported privileged_text_mode={mode!r}. "
            "Supported values: solution_answer (alias: solution+answer), answer_only."
        )
    return normalized


def build_teacher_privileged_text(answer: str, solution: str, mode: str) -> tuple[str, str]:
    mode = normalize_privileged_text_mode(mode)
    answer = str(answer).strip()
    solution = str(solution or "").strip()
    if mode == "solution_answer":
        if not solution:
            raise ValueError("privileged_text_mode=solution_answer requires a non-empty GT solution")
        return f"{solution}\n\nCorrect final answer: {answer}", mode
    return f"Correct final answer: {answer}", mode


def build_teacher_messages(
    problem: str, answer: str, solution: str, mode: str,
    transition_prompt: Optional[str] = None,
    framing_prompt: Optional[str] = None,
    thinking: bool = False,
) -> tuple[str, list[dict], str]:
    """Build the fixed teacher message from the reference solution."""
    if framing_prompt is None:
        framing_prompt = "Here is a reference solution to this problem:"
    if transition_prompt is None:
        transition_prompt = TEACHER_TRANSITION_PROMPT
    problem = strip_legacy_math_prompt(problem)
    privileged_text, effective_mode = build_teacher_privileged_text(answer=answer, solution=solution, mode=mode)
    del thinking
    user_content = (
        f"Problem: {problem}\n\n"
        f"{framing_prompt}\n"
        "=== Reference Solution Begin ===\n"
        f"{privileged_text}\n"
        "=== Reference Solution End ==="
        f"{transition_prompt}\n"
        f"{BOXED_ANSWER_INSTRUCTION}"
    )
    messages = [{"role": "user", "content": user_content}]
    return privileged_text, messages, effective_mode


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer using the official OPSD logic."""
    if text is None:
        return None

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def _normalize_fallback_string(text: str) -> str:
    return str(text).replace("$", "").replace(" ", "").lower().strip()


def _strip_answer_wrappers(text: str) -> str:
    text = str(text).strip()
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$").strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text.replace(r"\left", "").replace(r"\right", "").strip()


def _canonical_simple_number(text: str) -> Optional[Decimal]:
    text = _strip_answer_wrappers(text)
    text = _DEGREE_SUFFIX_RE.sub("", text.replace(" ", "").replace(",", "").replace("\u2212", "-"))
    if not _SIMPLE_NUMBER_RE.fullmatch(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _simple_numeric_answers_match(predicted: str, ground_truth: str) -> Optional[bool]:
    pred_number = _canonical_simple_number(predicted)
    gt_number = _canonical_simple_number(ground_truth)
    if pred_number is None or gt_number is None:
        return None
    return pred_number == gt_number


def grade_boxed_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Grade a boxed answer using official OPSD math_verify logic with string fallback."""
    if predicted is None:
        return False

    numeric_match = _simple_numeric_answers_match(predicted, ground_truth)
    if numeric_match is not None:
        return numeric_match

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ModuleNotFoundError(
            "math_verify is required for OPSD-aligned grading. Install `math-verify` in the training env."
        ) from exc

    try:
        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = ground_truth if "$" in str(ground_truth) else f"${ground_truth}$"
        pred_parsed = parse(pred_text, fallback_mode="no_fallback")
        gt_parsed = parse(gt_text, fallback_mode="no_fallback")
        return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        return _normalize_fallback_string(predicted) == _normalize_fallback_string(ground_truth)

# --------------------------------------------------------------------- #
# Exemplar-guided teaching (EGT): privileged context = a solved neighbor
# --------------------------------------------------------------------- #
#
# OPSD conditions the teacher on the reference solution of *the same* problem,
# which hands it the target answer. EGT instead shows a *different but
# related* problem together with its reference solution: the teacher never
# sees the answer it is supposed to supervise, so the dense signal carries
# transferable strategy instead of the answer itself.
#
# The scaffolding (Problem header, === Reference Solution Begin/End ===, final
# boxed instruction) is deliberately identical to build_teacher_messages so the
# two conditions differ only in the *content* of the reference block.

EGT_FRAMING_PROMPT = (
    "Here is a reference solution to a DIFFERENT but related problem. It is not the "
    "problem you must solve, and its final answer is NOT the answer to your problem "
    "-- use it only as a guide to the strategies and techniques that transfer:"
)

EGT_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you understand the "
    "reasoning behind each step, then solve the original problem with whatever "
    "transfers from it.\n"
)


def build_egt_privileged_text(exemplar_problem: str, exemplar_solution: str) -> str:
    """Compose the reference block body for the EGT teacher."""
    exemplar_problem = strip_legacy_math_prompt(exemplar_problem)
    return (
        f"Related problem: {exemplar_problem}\n\n"
        f"Solution to the related problem:\n{str(exemplar_solution).strip()}"
    )


def build_egt_teacher_messages(
    problem: str,
    exemplar_problem: str,
    exemplar_solution: str,
    framing_prompt: Optional[str] = None,
    transition_prompt: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """Build the EGT teacher user message. Returns (privileged_text, messages)."""
    problem = strip_legacy_math_prompt(problem)
    privileged_text = build_egt_privileged_text(exemplar_problem, exemplar_solution)
    user_content = (
        f"Problem: {problem}\n\n"
        f"{framing_prompt or EGT_FRAMING_PROMPT}\n"
        "=== Reference Solution Begin ===\n"
        f"{privileged_text}\n"
        "=== Reference Solution End ==="
        f"{transition_prompt or EGT_TRANSITION_PROMPT}\n"
        f"{BOXED_ANSWER_INSTRUCTION}"
    )
    return privileged_text, [{"role": "user", "content": user_content}]


def build_plain_teacher_messages(problem: str) -> tuple[str, list[dict]]:
    """Non-privileged teacher message (used as the EGT retrieval-miss fallback)."""
    problem = strip_legacy_math_prompt(problem)
    return "", [{"role": "user", "content": f"Problem: {problem}\n\n{BOXED_ANSWER_INSTRUCTION}"}]

