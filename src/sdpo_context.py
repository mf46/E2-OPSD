"""Pure helpers for official-style SDPO sibling privileged contexts."""

import re


_THINKING_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_demonstration_thinking(text: str) -> str:
    """Remove complete thinking spans with the same regex semantics as official SDPO."""
    return _THINKING_RE.sub("", str(text or ""))


def select_sibling_solutions(
    uids,
    rewards,
    responses,
    success_threshold: float = 0.5,
    exclude_self: bool = True,
    strip_thinking: bool = True,
):
    """Select the first eligible successful sibling for every rollout."""
    if not (len(uids) == len(rewards) == len(responses)):
        raise ValueError("uids, rewards, and responses must have identical lengths")

    groups = {}
    normalized_uids = [str(uid) for uid in uids]
    for idx, uid in enumerate(normalized_uids):
        groups.setdefault(uid, []).append(idx)

    successful_by_uid = {
        uid: [idx for idx in indices if float(rewards[idx]) >= float(success_threshold)]
        for uid, indices in groups.items()
    }
    source_indices = []
    raw_solutions = []
    cleaned_solutions = []
    valid = []
    for idx, uid in enumerate(normalized_uids):
        candidates = successful_by_uid[uid]
        if exclude_self:
            candidates = [candidate for candidate in candidates if candidate != idx]
        source_idx = candidates[0] if candidates else None
        raw_solution = str(responses[source_idx]) if source_idx is not None else ""
        cleaned_solution = (
            strip_demonstration_thinking(raw_solution) if strip_thinking else raw_solution.strip()
        )
        is_valid = source_idx is not None and bool(cleaned_solution)
        source_indices.append(source_idx if is_valid else None)
        raw_solutions.append(raw_solution if is_valid else "")
        cleaned_solutions.append(cleaned_solution if is_valid else "")
        valid.append(is_valid)

    return {
        "groups": groups,
        "successful_by_uid": successful_by_uid,
        "source_indices": source_indices,
        "raw_solutions": raw_solutions,
        "cleaned_solutions": cleaned_solutions,
        "valid": valid,
    }
