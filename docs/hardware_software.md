# Hardware and software of the measurements

## Main platform (all 4090 numbers in the paper)

| item | value |
|---|---|
| CPU | Intel Core i9-14900KF (24 cores, 32 threads) |
| RAM | 62 GB |
| GPU | 1 x NVIDIA GeForce RTX 4090, 24 GB; driver 535.309.01 |
| OS / kernel | Ubuntu 24.04.1 LTS, Linux 6.11 |
| Python | 3.11 |
| Key packages | torch 2.9.1+cu126, triton 3.5.1, diffusers 0.35.2, transformers 4.56.2, peft 0.17.1, accelerate 1.11.0, numpy 2.2.6, pandas 3.1.1, scipy 1.17.1, matplotlib 3.11.2, pytest 9.2.0 (full list: `env/lock_p24_diff.txt`) |

The CUDA runtime comes with the torch wheel (cu126); no system CUDA toolkit is needed for the CPU-only parts.

## Timing protocol (summary)

* Formal timing runs use the GPU exclusively (no other process on it) in blocks of at most two hours.
* The host was a shared machine: formal timing was pinned to the performance cores (`taskset -c 0-15`), required a load
  average of at most 16 and at least 25 GB of available memory; runs outside these conditions are not used for main results.
* One process is the statistical unit; arms are interleaved across processes; every one-time cost (compile, capture,
  cache load) is charged to total service time unless the paper states otherwise (the offline AOT build is reported separately).
* Equivalence of outputs is judged against an fp32 eager reference (tolerance frozen before the comparison: RMSE ratio
  <= 2 and LPIPS <= 0.02; `code/tests/tolerances.json`).

## Other platforms

An NVIDIA H200 node was used only for one replay of the AOT result (reported separately in the paper). Platforms are never mixed
in one comparison.

## Reproducibility notes

* Across environments the simulator output must agree numerically (all columns except wall-clock columns); figure PNG bytes
  are only identical in the original environment.
* Large single pinned-host allocations (> 1 GiB) were observed to fail intermittently on the measurement host; see
  `docs/capture_constraints.md`, entries 18 to 18c.
