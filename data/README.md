# Data

Files, sizes and SHA-256 hashes are listed in [manifest.json](manifest.json).

```bash
python scripts/download_data.py            # download the training file and verify both files
python scripts/download_data.py --verify   # verify without network access
```

| File | Source |
| --- | --- |
| `verl/deepmath_filtered_level5_7/train.parquet` | [Leyiii/RLCSD](https://huggingface.co/datasets/Leyiii/RLCSD/tree/33d7de919af5b03257ff92c30303fddf9afdda4a), from [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |
| `verl/amc23+aime24+aime25+hmmt25/val.parquet` (bundled) | AMC23, AIME24, AIME25: [Leyiii/RLCSD](https://huggingface.co/datasets/Leyiii/RLCSD/tree/33d7de919af5b03257ff92c30303fddf9afdda4a), compiled from [AI-MO/aimo-validation-amc](https://huggingface.co/datasets/AI-MO/aimo-validation-amc) and [math-ai/aime24](https://huggingface.co/datasets/math-ai/aime24). HMMT25: [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025/tree/6fdc4277120810ff75aa22d2d5489b91f7a262a1), converted to the veRL format without changing questions or answers. |

## Licenses

Dataset licenses are separate from the MIT code license.

- RLCSD data and DeepMath-103K: MIT.
- AMC and AIME problems compiled by AI-MO and math-ai: [Apache License 2.0](licenses/Apache-2.0.txt).
- HMMT February 2025 (Harvard–MIT Mathematics Tournament, distributed by [MathArena](https://matharena.ai/)): [CC BY-NC-SA 4.0](licenses/CC-BY-NC-SA-4.0.txt). Neither MathArena nor HMMT endorses this project.
