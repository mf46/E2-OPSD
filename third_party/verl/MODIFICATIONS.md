# Modifications to veRL

This directory is a modified copy of [veRL](https://github.com/volcengine/verl)
at commit [`59f53cc4c92dee9609701bbb10d71238bd9ca86e`](https://github.com/volcengine/verl/commit/59f53cc4c92dee9609701bbb10d71238bd9ca86e),
licensed under the Apache License 2.0 (see [LICENSE](LICENSE) and [Notice.txt](Notice.txt)).
Modules not used by E²-OPSD, together with tests, examples and docs, were removed.

## Modified files

Each modified file carries the notice
`Modified by the E2-OPSD authors, 2026; see third_party/verl/MODIFICATIONS.md.`

- `pyproject.toml`
- `requirements.txt`
- `setup.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
- `verl/trainer/config/actor/actor.yaml`
- `verl/trainer/config/data/legacy_data.yaml`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/chat_template.py`
- `verl/utils/dataset/rl_dataset.py`
- `verl/utils/model.py`
- `verl/utils/torch_functional.py`
- `verl/utils/tracking.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/config/actor.py`
- `verl/workers/config/rollout.py`
- `verl/workers/engine_workers.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `verl/workers/utils/losses.py`

## Added files

- `verl/utils/canonical_eval.py`
- `verl/utils/reproducibility.py`
