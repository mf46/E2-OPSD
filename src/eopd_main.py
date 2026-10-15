"""OPSD + EOPD entry: shared runtime, best Hugging Face checkpoint only."""
import hydra
import ray

from src import hf_best_checkpoint
from src.self_distill_main import SelfDistillTaskRunner, _ensure_legacy_worker_impl
from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
from verl.utils.device import auto_set_device


class BaselineTaskRunner(SelfDistillTaskRunner):
    def run(self, config):
        assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "eopd"
        hf_best_checkpoint.install(config)
        return super().run(config)


@hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = _ensure_legacy_worker_impl(migrate_legacy_reward_impl(config))
    # Import by canonical module name so Ray does not serialize __main__ globals.
    from src.eopd_main import BaselineTaskRunner as ImportableTaskRunner

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ImportableTaskRunner))


if __name__ == "__main__":
    main()
