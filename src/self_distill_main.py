"""
Entry point for self-distillation training using verl's infrastructure.

Self-distillation flow per rollout step:
1. Student rollout: vLLM generates responses (verl standard)
2. Privileged text extraction: select the configured dataset or rollout context
3. Teacher forward: compute log probs with privileged context on the same response tokens
4. Actor update with distillation loss

Usage:
    python -m src.self_distill_main <verl hydra config overrides>
"""

import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "verl"))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import open_dict

from src.opsd_format import (
    build_rollout_messages,
    build_teacher_messages,
    extract_boxed_answer,
    normalize_privileged_text_mode,
    strip_legacy_math_prompt)
from src.opsd_format import (
    build_egt_teacher_messages,
    build_plain_teacher_messages)
from src.egt_retrieval import (
    ExemplarRetriever,
    build_corpus_from_parquet,
    normalize_question,
    truncate_chars)
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from src.sdpo_context import select_sibling_solutions

DISTILL_LOSS_MODES = {"e2opsd", "opsd", "eopd", "sdpo", "rlsd"}
MATH_DATA_SOURCES = {
    "deepmath_filtered",
    "deepmath_filtered_level5_7",
    "deepmath_filtered_level6_8",
    "deepmath_filtered_level7_10",
    "amc23",
    "aime24",
    "aime25",
    "hmmt25",
}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for segment in content:
            if isinstance(segment, dict):
                if segment.get("type") == "text":
                    parts.append(str(segment.get("text", "")))
                elif "content" in segment:
                    parts.append(str(segment.get("content", "")))
            elif segment is not None:
                parts.append(str(segment))
        return " ".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _raw_prompt_to_problem(raw_prompt):
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, list):
        user_parts = []
        fallback_parts = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            text = _content_to_text(msg.get("content", ""))
            if not text:
                continue
            fallback_parts.append(text)
            if msg.get("role") == "user":
                user_parts.append(text)
        parts = user_parts or fallback_parts
        return strip_legacy_math_prompt("\n".join(parts).strip())

    if raw_prompt is None:
        return ""
    return strip_legacy_math_prompt(str(raw_prompt).strip())


def _policy_loss_cfg(self):
    return self.config.actor_rollout_ref.actor.policy_loss


def _rollout_custom_cfg(self):
    rollout_cfg = self.config.actor_rollout_ref.get("rollout", {})
    custom_cfg = rollout_cfg.get("custom", None)
    return custom_cfg if custom_cfg is not None else {}


def _cfg_get(self, key, default):
    return _policy_loss_cfg(self).get(key, default)


def _custom_cfg_get(self, key, default):
    custom_cfg = _rollout_custom_cfg(self)
    if key in custom_cfg:
        return custom_cfg.get(key)
    return _cfg_get(self, key, default)


def _cfg_bool(self, key, default=False):
    value = _cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _custom_cfg_bool(self, key, default=False):
    value = _custom_cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _teacher_total_token_budget(self) -> int:
    actor_cfg = self.config.actor_rollout_ref.actor
    budget = actor_cfg.get("ppo_max_token_len_per_gpu", None)
    if budget is not None:
        return int(budget)
    return int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)


def _teacher_chat_template_kwargs(self):
    kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
    kwargs["enable_thinking"] = _custom_cfg_bool(self, "teacher_enable_thinking", True)
    return kwargs


def _collapse_repeated_to_batch(values, batch_size):
    if values is None:
        return None

    values = list(values)
    if len(values) == batch_size:
        return values
    if batch_size <= 0 or len(values) % batch_size != 0:
        raise AssertionError(f"Cannot align {len(values)} items to batch size {batch_size}")

    repeat_factor = len(values) // batch_size
    return [values[i * repeat_factor] for i in range(batch_size)]


def _extract_problems_answers_solutions(self, batch: DataProto, raw_prompts=None, reward_models=None, extra_infos=None):
    """Extract problem texts, GT answers, and GT solutions from the dataset batch."""
    batch_size = len(batch)
    if raw_prompts is None:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if reward_models is None:
        reward_models = batch.non_tensor_batch.get("reward_model")
    if extra_infos is None:
        extra_infos = batch.non_tensor_batch.get("extra_info")

    raw_prompts = _collapse_repeated_to_batch(raw_prompts, batch_size) if raw_prompts is not None else None
    reward_models = _collapse_repeated_to_batch(reward_models, batch_size) if reward_models is not None else None
    extra_infos = _collapse_repeated_to_batch(extra_infos, batch_size) if extra_infos is not None else None

    problems = []
    ground_truths = []
    solutions = []
    for i in range(batch_size):
        raw_prompt = raw_prompts[i] if raw_prompts is not None else None
        reward_model = reward_models[i] if reward_models is not None else {}
        answer = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(answer))

        extra_info = extra_infos[i] if extra_infos is not None else {}
        problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
        if not problem:
            problem = _raw_prompt_to_problem(raw_prompt)
        if not problem and "prompts" in batch.batch:
            problem = strip_legacy_math_prompt(
                self.tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=True).strip()
            )
        problems.append(problem)

        solution = extra_info.get("solution", "") if isinstance(extra_info, dict) else ""
        solutions.append(_content_to_text(solution).strip())

    return problems, ground_truths, solutions


def _expand_to_rollout(values, batch_size, rollout_n):
    num_values = len(values)
    if num_values == batch_size:
        return values
    if num_values * rollout_n != batch_size:
        raise AssertionError(f"Mismatch: cached={num_values}, rollout_n={rollout_n}, batch={batch_size}")

    expanded = []
    for value in values:
        for _ in range(rollout_n):
            expanded.append(value)
    return expanded


# ------------------------------------------------------------------ #
# Exemplar-guided teaching (EGT): the teacher sees a solved neighboring problem
# ------------------------------------------------------------------ #

def _egt_enabled(self) -> bool:
    return _custom_cfg_bool(self, "egt", False)


def _egt_resolve_train_parquets(self):
    """Training parquet path(s); their reference solutions form the exemplar pool."""
    files = self.config.data.get("train_files", None)
    if files is None:
        return []
    if isinstance(files, str):
        return [files]
    return [str(f) for f in files]


def _get_egt_retriever(self):
    retriever = getattr(self, "_egt_retriever", None)
    if retriever is not None:
        return retriever

    neighbors_path = str(_custom_cfg_get(self, "egt_neighbors_path", "") or "")
    if not neighbors_path or not os.path.isfile(neighbors_path):
        raise ValueError(
            "egt=true requires egt_neighbors_path to point at the precomputed "
            f"neighbor JSON (got {neighbors_path!r}). Build it with "
            "scripts/build_similar_questions.py.")

    parquets = _egt_resolve_train_parquets(self)
    if not parquets:
        raise ValueError("EGT requires data.train_files to build the reference corpus")
    retriever = ExemplarRetriever(
        neighbors_path=neighbors_path,
        corpus=build_corpus_from_parquet(parquets),
        max_solutions_per_question=int(_custom_cfg_get(self, "egt_max_solutions_per_question", 8)),
        seed=int(self.config.data.get("seed", 42) or 42))

    self._egt_retriever = retriever
    print(f"[EGT] {retriever.corpus_size()} reference questions loaded", flush=True)
    return retriever


def _build_egt_teacher_payloads(self, problems, ground_truths, solutions):
    """Teacher prompts whose privileged context is a retrieved solved neighbor.

    `problems` is per-sample (batch of prompts x rollouts). The group slot,
    `index % rollout.n`, seeds each rollout's own draw; it is derived from the
    index rather than a shared RNG so the draw is reproducible regardless of
    batch ordering. When no neighbor has a reference solution, the teacher
    falls back to the plain problem prompt (no privileged context).
    """
    del ground_truths, solutions  # EGT never shows the current problem's answer.
    retriever = _get_egt_retriever(self)
    global_step = int(getattr(self, "global_steps", 0) or 0)
    problem_budget = int(_custom_cfg_get(self, "egt_problem_char_budget", 0))
    solution_budget = int(_custom_cfg_get(self, "egt_solution_char_budget", 0))

    rollout_n = int(self.config.actor_rollout_ref.rollout.n)
    per_sample = rollout_n > 0 and len(problems) % rollout_n == 0 and len(problems) >= rollout_n

    privileged_texts = []
    teacher_prompts = []
    effective_modes = []
    exemplar_problems = []
    exemplar_ranks = []
    exemplar_chars = []
    hits = 0
    for index, problem in enumerate(problems):
        group_slot = (index % rollout_n) if per_sample else 0
        exemplar = retriever.retrieve(problem, global_step=global_step, group_slot=group_slot)
        if exemplar is not None:
            hits += 1
            exemplar_chars.append(len(exemplar.solution))
            privileged_text, messages = build_egt_teacher_messages(
                problem=problem,
                exemplar_problem=truncate_chars(exemplar.problem, problem_budget),
                exemplar_solution=truncate_chars(exemplar.solution, solution_budget))
            effective_modes.append("egt_exemplar")
            exemplar_problems.append(exemplar.problem)
            exemplar_ranks.append(int(exemplar.rank))
        else:
            privileged_text, messages = build_plain_teacher_messages(problem)
            effective_modes.append("egt_miss_plain")
            exemplar_problems.append("")
            exemplar_ranks.append(-1)
        privileged_texts.append(privileged_text)
        teacher_prompts.append(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_teacher_chat_template_kwargs(self)))

    total = max(len(problems), 1)
    hit_fraction = hits / total
    min_hit = float(_custom_cfg_get(self, "egt_min_hit_fraction", 0.5))
    if hit_fraction < min_hit:
        raise ValueError(
            f"[EGT] retrieval hit fraction {hit_fraction:.3f} < egt_min_hit_fraction "
            f"{min_hit} at step {global_step} (corpus_questions={retriever.corpus_size()}, "
            f"neighbor_entries={len(retriever.neighbors)}). Refusing to train on "
            "mostly-unprivileged teacher prompts.")

    # Distinct exemplars per group: 1.0 means every rollout of every group saw
    # a different exemplar, 1/n means the group shared one. Rows are grouped
    # by problem text because veRL's sequence-length balancing reorders rows.
    distinct_fraction = 0.0
    if per_sample and hits:
        by_problem: dict[str, set] = {}
        row_counts: dict[str, int] = {}
        for problem, exemplar_problem in zip(problems, exemplar_problems):
            key = normalize_question(problem)
            row_counts[key] = row_counts.get(key, 0) + 1
            if exemplar_problem:
                by_problem.setdefault(key, set()).add(exemplar_problem)
        distinct = sum(len(refs) for refs in by_problem.values())
        distinct_fraction = distinct / max(sum(row_counts.values()), 1)

    self._egt_last_stats = {
        "egt/hit_fraction": hit_fraction,
        "egt/corpus_questions": float(retriever.corpus_size()),
        "egt/corpus_solutions": float(retriever.corpus_solutions()),
        "egt/mean_exemplar_chars": float(np.mean(exemplar_chars)) if exemplar_chars else 0.0,
        "egt/distinct_exemplar_fraction": distinct_fraction,
    }
    return {
        "privileged_texts": privileged_texts,
        "teacher_prompts": teacher_prompts,
        "effective_modes": effective_modes,
        "egt_exemplar_problems": exemplar_problems,
        "egt_exemplar_ranks": exemplar_ranks,
        "rationales": privileged_texts}


def _egt_post_teacher_hook(self, batch: DataProto, problems):
    """Collect exemplar retrieval metrics for the current update."""
    stats = dict(getattr(self, "_egt_last_stats", {}) or {})
    existing = getattr(self, "_teacher_step_metrics", None)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(stats)
    self._teacher_step_metrics = existing


def _build_teacher_context_payloads(self, problems, ground_truths, solutions):
    if _egt_enabled(self):
        # EGT: privileged context is a retrieved solved neighboring problem
        # instead of this problem's own reference solution.
        return _build_egt_teacher_payloads(self, problems, ground_truths, solutions)

    requested_mode = normalize_privileged_text_mode(_custom_cfg_get(self, "privileged_text_mode", "solution_answer"))
    privileged_texts = []
    effective_modes = []
    teacher_prompts = []

    if requested_mode == "solution_answer":
        missing_indices = [i for i, solution in enumerate(solutions) if not solution]
        if missing_indices:
            preview = ", ".join(str(i) for i in missing_indices[:8])
            if len(missing_indices) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                "privileged_text_mode=solution_answer requires every sample to have a non-empty GT solution, "
                f"but {len(missing_indices)}/{len(solutions)} samples in the current batch are missing it "
                f"(batch indices: {preview})."
            )

    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    for problem, answer, solution in zip(problems, ground_truths, solutions):
        privileged_text, messages, effective_mode = build_teacher_messages(
            problem=problem,
            answer=answer,
            solution=solution,
            mode=requested_mode,
            thinking=use_thinking)
        privileged_texts.append(privileged_text)
        effective_modes.append(effective_mode)
        teacher_prompts.append(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_teacher_chat_template_kwargs(self))
        )

    return {
        "privileged_texts": privileged_texts,
        "teacher_prompts": teacher_prompts,
        "effective_modes": effective_modes,
        "rationales": privileged_texts}


def _generate_rationales(self, batch: DataProto, gen_output: DataProto) -> dict | None:
    """Cache GT privileged text from the dataset batch for teacher forwarding."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode == "sdpo" and str(
        _custom_cfg_get(self, "privileged_solution_source", "dataset")
    ).strip().lower() == "sibling_success":
        # Official SDPO-style contexts can only be chosen after verifier
        # rewards identify successful siblings.
        return None
    if _egt_enabled(self):
        # EGT builds its teacher batch in _compute_teacher_log_probs, where the
        # batch is already expanded to one row per rollout: that is the only
        # place a group slot (index % rollout.n) is meaningful, and the rollout
        # dump reads those per-sample payloads. Retrieving here as well would
        # duplicate the work under a per-prompt indexing that does not apply.
        return None

    raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if raw_prompts is None:
        raw_prompts = gen_output.non_tensor_batch.get("raw_prompt")

    problems, ground_truths, solutions = self._extract_problems_answers_solutions(
        batch,
        raw_prompts=raw_prompts,
        reward_models=batch.non_tensor_batch.get("reward_model"),
        extra_infos=batch.non_tensor_batch.get("extra_info"))
    payloads = _build_teacher_context_payloads(self, problems, ground_truths, solutions)
    payloads.update({"problems": problems, "ground_truths": ground_truths, "solutions": solutions})
    return payloads


def _align_resp_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate the response dimension (dim=1) to match target_len."""
    current_len = tensor.shape[1]
    if current_len == target_len:
        return tensor
    if current_len < target_len:
        return F.pad(tensor, (0, target_len - current_len), value=0.0)
    return tensor[:, :target_len]


def _build_teacher_batch_from_prompts(self, teacher_prompts, responses, prefix="teacher") -> DataProto:
    """Pack teacher prompts and sampled responses into model-ready tensors."""
    batch_size = len(teacher_prompts)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    pad_id = self.tokenizer.pad_token_id or 0
    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]

    teacher_input_ids_list = []
    teacher_prompt_lens = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        prompt_ids = teacher_prompt_ids_list[i]
        response_ids = [token for token in responses[i].tolist() if token != pad_id]
        max_prompt_tokens = max(teacher_total_budget - len(response_ids), 0)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
        teacher_input_ids_list.append(prompt_ids + response_ids)
        teacher_prompt_lens.append(len(prompt_ids))

    max_prompt_len = max(teacher_prompt_lens)
    response_len = max(len(ids) - teacher_prompt_lens[i] for i, ids in enumerate(teacher_input_ids_list))
    total_len = max_prompt_len + response_len

    teacher_input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
    teacher_attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long)
    teacher_position_ids = torch.zeros((batch_size, total_len), dtype=torch.long)

    for i, ids in enumerate(teacher_input_ids_list):
        prompt_len = teacher_prompt_lens[i]
        prompt_offset = max_prompt_len - prompt_len
        response_ids = ids[prompt_len:]
        response_count = len(response_ids)

        teacher_input_ids[i, prompt_offset:max_prompt_len] = torch.tensor(ids[:prompt_len], dtype=torch.long)
        if response_count > 0:
            teacher_input_ids[i, max_prompt_len:max_prompt_len + response_count] = torch.tensor(
                response_ids, dtype=torch.long
            )

        teacher_attention_mask[i, prompt_offset:max_prompt_len + response_count] = 1
        teacher_position_ids[i, prompt_offset:max_prompt_len + response_count] = torch.arange(
            prompt_len + response_count
        )

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": teacher_input_ids,
            f"{prefix}_attention_mask": teacher_attention_mask,
            f"{prefix}_position_ids": teacher_position_ids}
    )


# ------------------------------------------------------------------ #
# Step 3: Compute teacher log probs (called during training phase)
# ------------------------------------------------------------------ #

def _record_teacher_prompt_length_metrics(self, batch, teacher_prompts, responses):
    """Record the same combined-budget truncation metadata for every teacher source."""
    pad_id = self.tokenizer.pad_token_id or 0
    prompt_ids = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]
    before = []
    after = []
    truncated_tokens = []
    truncated = []
    total_budget = _teacher_total_token_budget(self)
    for idx, ids in enumerate(prompt_ids):
        response_ids = [token for token in responses[idx].tolist() if token != pad_id]
        kept = min(len(ids), max(total_budget - len(response_ids), 0))
        removed = len(ids) - kept
        before.append(len(ids))
        after.append(kept)
        truncated_tokens.append(removed)
        truncated.append(1.0 if removed > 0 else 0.0)
    batch.non_tensor_batch["teacher_prompt_truncated"] = np.asarray(truncated, dtype=np.float32)
    batch.non_tensor_batch["teacher_prompt_truncated_tokens"] = np.asarray(
        truncated_tokens, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_before_trunc"] = np.asarray(before, dtype=np.float32)
    batch.non_tensor_batch["teacher_prompt_tokens_after_trunc"] = np.asarray(after, dtype=np.float32)


def _compute_sdpo_sibling_teacher_inputs(self, batch: DataProto) -> DataProto:
    """Build OPSD-format teacher prompts from non-self successful siblings."""
    batch_size = batch.batch.batch_size[0]
    uid_values = batch.non_tensor_batch.get("uid")
    if uid_values is None:
        raise ValueError("SDPO sibling-success contexts require uid in non_tensor_batch")
    if "token_level_scores" not in batch.batch.keys():
        raise ValueError("SDPO sibling-success contexts require token_level_scores after reward computation")

    problems, ground_truths, _ = self._extract_problems_answers_solutions(batch)
    responses = batch.batch["responses"]
    response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    reward_tensor = batch.batch["token_level_scores"]
    if reward_tensor.dim() > 1:
        rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
    else:
        rewards = reward_tensor.detach().cpu().tolist()
    uids = uid_values.tolist() if hasattr(uid_values, "tolist") else list(uid_values)

    threshold = float(_custom_cfg_get(self, "sdpo_success_reward_threshold", 0.5))
    exclude_self = _custom_cfg_bool(self, "sdpo_exclude_self_success", True)
    strip_thinking = _custom_cfg_bool(self, "sdpo_strip_sibling_thinking", True)
    selected = select_sibling_solutions(
        uids=uids,
        rewards=rewards,
        responses=response_texts,
        success_threshold=threshold,
        exclude_self=exclude_self,
        strip_thinking=strip_thinking,
    )

    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    teacher_prompts = []
    privileged_texts = []
    extracted_answers = []
    effective_modes = []
    valid = selected["valid"]
    for idx in range(batch_size):
        if valid[idx]:
            raw_solution = selected["raw_solutions"][idx]
            cleaned_solution = selected["cleaned_solutions"][idx]
            answer = (
                extract_boxed_answer(cleaned_solution)
                or extract_boxed_answer(raw_solution)
                or ground_truths[idx]
            )
            privileged_text, messages, effective_mode = build_teacher_messages(
                problem=problems[idx],
                answer=answer,
                solution=cleaned_solution,
                mode="solution_answer",
                    thinking=use_thinking,
            )
        else:
            # Official SDPO masks rows with no eligible demonstration. Keep a
            # plain prompt only to preserve batch shape for distributed forward.
            privileged_text, messages = build_plain_teacher_messages(problems[idx])
            answer = ""
            effective_mode = "plain_no_sibling"
        teacher_prompts.append(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_teacher_chat_template_kwargs(self),
            )
        )
        privileged_texts.append(privileged_text)
        extracted_answers.append(answer)
        effective_modes.append(effective_mode)

    valid_array = np.asarray(valid, dtype=np.float32)
    success_group_count = sum(bool(indices) for indices in selected["successful_by_uid"].values())
    group_count = len(selected["groups"])
    success_rollout_count = sum(
        len(indices) for indices in selected["successful_by_uid"].values()
    )
    self._teacher_step_metrics = {
        "sdpo/success_group_fraction": float(success_group_count / max(group_count, 1)),
        "sdpo/success_rollout_fraction": float(success_rollout_count / max(batch_size, 1)),
        "sdpo/distillable_sample_fraction": float(valid_array.mean()) if batch_size else 0.0,
        "sdpo/no_sibling_fraction": float(1.0 - valid_array.mean()) if batch_size else 0.0,
    }
    self._aligned_teacher_rollout_details = {
        "privileged_texts": privileged_texts,
        "teacher_prompts": teacher_prompts,
        "effective_modes": effective_modes,
        "sdpo_sibling_source_indices": selected["source_indices"],
        "sdpo_sibling_raw_responses": selected["raw_solutions"],
        "sdpo_sibling_solutions": selected["cleaned_solutions"],
        "sdpo_sibling_answers": extracted_answers,
        "sdpo_distillation_valid": valid,
    }

    _record_teacher_prompt_length_metrics(self, batch, teacher_prompts, responses)

    teacher_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_prompts,
        responses=responses,
        prefix="teacher",
    )
    mask_batch = DataProto.from_dict(
        tensors={"sdpo_distillation_mask": torch.tensor(valid, dtype=torch.float32)}
    )
    return teacher_batch.union(mask_batch)


def _compute_teacher_log_probs(self, batch: DataProto) -> DataProto | None:
    """Build teacher-side inputs from the method's configured privileged source."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    self._data_metric_overrides = {}
    if loss_mode not in DISTILL_LOSS_MODES:
        return None

    self._teacher_step_metrics = {}
    if loss_mode == "sdpo" and str(
        _custom_cfg_get(self, "privileged_solution_source", "dataset")
    ).strip().lower() == "sibling_success":
        return _compute_sdpo_sibling_teacher_inputs(self, batch)

    batch_size = batch.batch.batch_size[0]
    problems, ground_truths, solutions = self._extract_problems_answers_solutions(batch)
    aligned_context = _build_teacher_context_payloads(self, problems, ground_truths, solutions)
    self._aligned_teacher_rollout_details = aligned_context
    if _egt_enabled(self):
        _egt_post_teacher_hook(self, batch, problems)
    teacher_prompts = aligned_context["teacher_prompts"]

    if len(teacher_prompts) != batch_size:
        rollout_n = self.config.actor_rollout_ref.rollout.n
        teacher_prompts = _expand_to_rollout(teacher_prompts, batch_size, rollout_n)

    responses = batch.batch["responses"]
    _record_teacher_prompt_length_metrics(self, batch, teacher_prompts, responses)
    teacher_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_prompts,
        responses=responses,
        prefix="teacher")

    return teacher_batch


def _teacher_context_metrics(self) -> dict:
    """Return per-step teacher-context metrics when available."""
    metrics = getattr(self, "_teacher_step_metrics", {})
    self._teacher_step_metrics = {}
    return dict(metrics)


def _get_data_metric_overrides(self) -> dict:
    """Return per-step metric overrides that should win over post-filter stats."""
    metrics = getattr(self, "_data_metric_overrides", {})
    self._data_metric_overrides = {}
    return dict(metrics)


_ORIGINAL_RLHF_BUILD_MESSAGES = None


def _patched_rlhf_build_messages(self, example: dict):
    """Format stored raw problems into official OPSD train/val rollout prompts."""
    messages = _ORIGINAL_RLHF_BUILD_MESSAGES(self, example)

    data_source = str(example.get("data_source", "")).lower()
    if data_source not in MATH_DATA_SOURCES:
        return messages

    extra_info = example.get("extra_info") or {}
    problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
    if not problem:
        problem = _raw_prompt_to_problem(messages)

    # Read thinking_system_prompt from data config (self = RLHFDataset).
    use_thinking = False
    cfg = getattr(self, "config", None)
    if cfg is not None:
        val = cfg.get("thinking_system_prompt", False)
        use_thinking = str(val).strip().lower() in ("1", "true", "yes", "on") if isinstance(val, str) else bool(val)
    return build_rollout_messages(problem, data_source=data_source, thinking=use_thinking)


def _save_train_rollout(self, batch: DataProto):
    """Save train rollout details with teacher context aligned to the current batch order."""
    log_dir = self._get_rollout_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"train_step_{self.global_steps}.jsonl")

    batch_size = batch.batch.batch_size[0]
    prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    score_tensor = batch.batch["token_level_scores"]
    if score_tensor.dim() > 1:
        scores = score_tensor.sum(dim=-1).cpu().tolist()
    else:
        scores = score_tensor.cpu().tolist()

    ground_truths = []
    for i in range(batch_size):
        item = batch[i]
        reward_model = item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(ground_truth))

    aligned_context = getattr(self, "_aligned_teacher_rollout_details", None)
    privileged_texts = aligned_context.get("privileged_texts", []) if aligned_context is not None else []
    teacher_prompts = aligned_context.get("teacher_prompts", []) if aligned_context is not None else []
    effective_modes = aligned_context.get("effective_modes", []) if aligned_context is not None else []
    egt_exemplar_problems = aligned_context.get("egt_exemplar_problems", []) if aligned_context is not None else []
    egt_exemplar_ranks = aligned_context.get("egt_exemplar_ranks", []) if aligned_context is not None else []
    sdpo_source_indices = aligned_context.get("sdpo_sibling_source_indices", []) if aligned_context is not None else []
    sdpo_raw_responses = aligned_context.get("sdpo_sibling_raw_responses", []) if aligned_context is not None else []
    sdpo_solutions = aligned_context.get("sdpo_sibling_solutions", []) if aligned_context is not None else []
    sdpo_answers = aligned_context.get("sdpo_sibling_answers", []) if aligned_context is not None else []
    sdpo_valid = aligned_context.get("sdpo_distillation_valid", []) if aligned_context is not None else []

    def _batch_metadata(name):
        values = batch.non_tensor_batch.get(name)
        if values is None:
            return []
        return values.tolist() if hasattr(values, "tolist") else list(values)

    teacher_prompt_truncated = _batch_metadata("teacher_prompt_truncated")
    teacher_prompt_truncated_tokens = _batch_metadata("teacher_prompt_truncated_tokens")
    teacher_prompt_tokens_before = _batch_metadata("teacher_prompt_tokens_before_trunc")
    teacher_prompt_tokens_after = _batch_metadata("teacher_prompt_tokens_after_trunc")

    lines = []
    for i in range(batch_size):
        entry = {
            "step": self.global_steps,
            "sample_idx": i,
            "prompt": prompts[i],
            "response": responses[i],
            "score": scores[i],
            "ground_truth": ground_truths[i]}
        if i < len(privileged_texts):
            entry["privileged_text"] = privileged_texts[i]
        if i < len(teacher_prompts):
            entry["teacher_prompt"] = teacher_prompts[i]
        if i < len(effective_modes):
            entry["privileged_text_mode"] = effective_modes[i]
        if i < len(egt_exemplar_problems):
            entry["egt_exemplar_problem"] = egt_exemplar_problems[i]
            entry["egt_exemplar_rank"] = egt_exemplar_ranks[i] if i < len(egt_exemplar_ranks) else -1
        if i < len(sdpo_source_indices):
            entry["sdpo_sibling_source_idx"] = sdpo_source_indices[i]
            entry["sdpo_sibling_raw_response"] = sdpo_raw_responses[i]
            entry["sdpo_sibling_solution"] = sdpo_solutions[i]
            entry["sdpo_sibling_answer"] = sdpo_answers[i]
            entry["sdpo_distillation_valid"] = bool(sdpo_valid[i])
        if i < len(teacher_prompt_truncated):
            entry["teacher_prompt_truncated"] = bool(teacher_prompt_truncated[i])
            entry["teacher_prompt_truncated_tokens"] = int(teacher_prompt_truncated_tokens[i])
            entry["teacher_prompt_tokens_before_trunc"] = int(teacher_prompt_tokens_before[i])
            entry["teacher_prompt_tokens_after_trunc"] = int(teacher_prompt_tokens_after[i])
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved train rollout details to {filename} ({batch_size} samples)")

    return {}


# ------------------------------------------------------------------ #
# Apply monkey patches
# ------------------------------------------------------------------ #

def _apply_patches():
    global _ORIGINAL_RLHF_BUILD_MESSAGES
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if _ORIGINAL_RLHF_BUILD_MESSAGES is None:
        _ORIGINAL_RLHF_BUILD_MESSAGES = RLHFDataset._build_messages
    RLHFDataset._build_messages = _patched_rlhf_build_messages

    RayPPOTrainer._extract_problems_answers_solutions = _extract_problems_answers_solutions
    RayPPOTrainer._generate_rationales = _generate_rationales
    RayPPOTrainer._compute_teacher_log_probs = _compute_teacher_log_probs
    RayPPOTrainer._teacher_context_metrics = _teacher_context_metrics
    RayPPOTrainer._get_data_metric_overrides = _get_data_metric_overrides
    RayPPOTrainer._save_train_rollout = _save_train_rollout


_apply_patches()


def _ensure_legacy_worker_impl(config):
    worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    if worker_impl == "disable":
        with open_dict(config):
            config.trainer.use_legacy_worker_impl = "enable"
        print(
            "[Self-Distill] Overriding trainer.use_legacy_worker_impl=disable -> enable. "
            "Privileged teacher prompting and full-logit distillation are currently implemented "
            "only in the legacy actor worker path."
        )
    return config


# ------------------------------------------------------------------ #
# Custom TaskRunner that applies patches inside the Ray actor
# ------------------------------------------------------------------ #

from verl.trainer.main_ppo import TaskRunner


def _validate_complete_e2opsd(config):
    policy = config.actor_rollout_ref.actor.policy_loss
    if policy.get("loss_mode", "vanilla") != "e2opsd":
        return
    custom = config.actor_rollout_ref.rollout.get("custom", {})
    if not custom.get("egt", False):
        raise ValueError("E²-OPSD requires the EGT teacher context (egt: true)")
    if policy.get("teacher_mode", "fixed") != "frozen":
        raise ValueError("E²-OPSD uses the frozen initial teacher (teacher_mode: frozen)")


class SelfDistillTaskRunner(TaskRunner):
    def run(self, config):
        from src.self_distill_main import _apply_patches as apply_sd_patches

        _validate_complete_e2opsd(config)
        apply_sd_patches()
        return super().run(config)


if __name__ == "__main__":
    import hydra
    import ray
    from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
    from verl.utils.device import auto_set_device

    @hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def main(config):
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        config = _ensure_legacy_worker_impl(config)
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SelfDistillTaskRunner))

    main()
