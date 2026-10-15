# Copyright 2026 The E2-OPSD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small reproducibility helpers shared by trainer and rollout code."""

from __future__ import annotations

import hashlib
from typing import Any


def derive_trajectory_seed(
    base_seed: int,
    *,
    step: int,
    sample_index: Any,
    rollout_n: int,
    validate: bool,
    data_source: Any = "",
    manifest_hash: str = "",
    eval_uid: Any = None,
) -> int:
    """Derive a stable vLLM request seed from an experiment base seed.

    Validation intentionally ignores ``step``. Every checkpoint therefore
    uses the same random stream for a given benchmark item and rollout number.
    Training includes ``step`` so revisiting a sample does not replay the same
    completion. ``hashlib`` avoids Python's process-salted ``hash``.
    """

    # Validation seeds are tied to the evaluation manifest and UID whenever
    # those are supplied; otherwise the seed is derived from the step below.
    if validate and manifest_hash and eval_uid is not None:
        from verl.utils.canonical_eval import derive_eval_seed

        return derive_eval_seed(int(base_seed), str(manifest_hash), eval_uid, int(rollout_n))
    effective_step = 0 if validate else int(step)
    identity = "\x1f".join(
        (
            str(int(base_seed)),
            str(effective_step),
            str(data_source),
            str(sample_index),
            str(int(rollout_n)),
            "val" if validate else "train",
        )
    )
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF
