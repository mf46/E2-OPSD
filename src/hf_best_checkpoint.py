"""Best-checkpoint retention in Hugging Face format for the EOPD and RLSD entries.

These baselines keep only the best validation checkpoint, exported as
Hugging Face weights. The trainer saves each scheduled checkpoint into a
staging directory; after validation it promotes the best one to
``hf_step{N}`` and removes the rest.
"""
import json
import shutil
from pathlib import Path

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _save_hf_checkpoint(self):
    root = Path(self.config.trainer.default_local_dir)
    staging = root / f".hf_staging_{self.global_steps}"
    assert not staging.exists(), str(staging)
    self.actor_rollout_wg.save_checkpoint(
        str(staging / "actor"), None, self.global_steps, max_ckpt_to_keep=None)


def _keep_best_hf_checkpoint(self, best_step, latest_step):
    from safetensors import safe_open

    # Check that the newly staged export is complete before touching anything.
    root = Path(self.config.trainer.default_local_dir)
    staging = root / f".hf_staging_{latest_step}"
    source = staging / "actor" / "huggingface"
    assert (source / "config.json").is_file()
    shards = list(source.glob("*.safetensors"))
    assert shards
    index = source / "model.safetensors.index.json"
    if index.exists():
        weight_files = set(json.loads(index.read_text())["weight_map"].values())
        assert all((source / name).is_file() for name in weight_files)
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            assert len(handle.keys()) > 0

    # Promote the staged export if it is the new best, then drop everything else.
    target = root / f"hf_step{best_step}"
    if int(best_step) == int(latest_step):
        assert not target.exists()
        source.rename(target)
    assert target.is_dir(), str(target)
    shutil.rmtree(staging)
    for old in root.glob("hf_step*"):
        if old != target and old.is_dir() and not old.is_symlink() and old.name[len("hf_step"):].isdigit():
            shutil.rmtree(old)

    record = {
        "best_step": int(best_step),
        "path": str(target),
        "format": "huggingface",
        "resume_supported": False,
    }
    (root / "best_hf_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")


def install(config):
    """Validate the checkpoint settings and patch the trainer to keep the best HF export."""
    assert config.trainer.save_best_only and not config.trainer.save_best_only_keep_latest
    assert list(config.actor_rollout_ref.actor.checkpoint.save_contents) == ["hf_model"]
    RayPPOTrainer._save_checkpoint = _save_hf_checkpoint
    RayPPOTrainer._prune_checkpoints_to_best_and_latest = _keep_best_hf_checkpoint
