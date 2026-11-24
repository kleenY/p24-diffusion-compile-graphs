# Data, models and licenses

Nothing in this repository is a model weight, an image, or a copy of a third-party dataset except the small
prompt text described below.

## DiffusionDB (workload prompts and shapes)

* Source: `poloclub/diffusiondb` (the 2M subset metadata). Only the metadata table is used (prompt, width, height,
  user name); no image is read. License: CC0 1.0 (public-domain dedication) as stated by the dataset; checked on 2026-11-18.
* Not distributed: the metadata file (`data/diffusiondb/metadata.parquet`). Download it from the dataset page and place it
  there; the expected SHA-256 is in `docs/workload_declaration.md` section 1.
* Distributed: the A/B split table (`results/S2/workload/split_A_B.parquet`: row index and half only), generated traces
  with N <= 1000 requests, which contain the prompt text of the drawn rows **unmodified**. The prompts are third-party
  user text (some contain non-English characters); they are the only place in the repository where such text appears.
* Traces with N > 1000 are not distributed. They are rebuilt deterministically from row ids and seeds
  (`code/scripts/gen_traces_S3.py`) and checked against `results/S3/workload/sha256_manifest.txt`.
* The A/B split, resolution mapping and request generator are declared in `docs/workload_declaration.md`.
  The batch-size and guidance distributions are declared assumptions, not measurements.

## GenEval (quality check)

* The quality check uses the GenEval prompt metadata and the GenEval evaluator on generated images; the generated images,
  the evaluator weights and the prompt file are not distributed. Fetch the evaluator from its public repository and put
  the prompt file at the path given in `code/scripts/analyze_geneval_S4.py`. GenEval (github.com/djghosh13/geneval) is
  released under the MIT License (checked 2026-11-23).
* Distributed: the aggregated scores and confidence intervals (`results/S4/geneval/analysis/`).

## Model weights (not distributed)

| model | repository | revision | license note |
|---|---|---|---|
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | CreativeML OpenRAIL-M (model card metadata, checked 2026-11-23); weights not redistributed |
| LCM-LoRA SD1.5 | `latent-consistency/lcm-lora-sdv1-5` | `cf2fced511dbe7e26c8d1d397e728fbab875db4b` | OpenRAIL++ (model card metadata, checked 2026-11-23); weights not redistributed |
| SD-Turbo (mechanism check X10) | `stabilityai/sd-turbo` | `b261bac6fd2cf515557d5d0707481eafa0485ec2` | Stability AI Community License; only the revision is given, the weights are not redistributed |

The fused SD1.5 + LCM-LoRA model is rebuilt locally from the two pinned revisions; `index/model_revisions.json` lists the
SHA-256 of every file of the fused model (`fused_files_sha256`) and `index/fused_weights_sha_measured.json` the measured
weight hashes, so a rebuilt copy can be compared.
