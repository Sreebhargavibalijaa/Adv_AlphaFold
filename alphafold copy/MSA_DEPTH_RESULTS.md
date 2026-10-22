# MSA Depth Benchmark

AlphaFold2 inference via ColabFold 1.6.3 on an Apple Silicon CPU, using the
171-residue target sequence in `benchmark_msa_depth.py`. This uses the installed
AlphaFold2 implementation, not the upstream repository's database pipeline.

The unfiltered MMseqs2 search returned 64 distinct aligned sequences. Depth
includes the query. Inputs are nested prefixes of the same deduplicated alignment;
no rows were repeated or fabricated to reach a requested depth.

| Requested depth | Actual input depth | Mean pLDDT | pTM | Mean PAE (angstrom) | Prediction time (s) | Wall time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 16 | 73.053 | 0.67 | 12.865 | 211.3 | 232.375 |
| 32 | 32 | 80.936 | 0.73 | 11.617 | 223.4 | 274.125 |
| 64 | 64 | 81.888 | 0.73 | 11.343 | 222.1 | 230.204 |
| 128 | unavailable | - | - | - | - | - |
| 256 | unavailable | - | - | - | - | - |

Confidence metrics are extracted from the score JSON files. pTM is rounded by
ColabFold in these files. Higher pLDDT/pTM and lower PAE indicate higher predicted
confidence, not experimentally measured accuracy. No reference structure was
provided, so reference-based RMSD, TM-score, and lDDT were not evaluated.

All runs use `alphafold2_ptm`, model 1, seed 0, three recycles, no templates and
no relaxation. The clustered MSA limit equals the input depth; the extra-MSA
allocation is one row, with no additional homologs supplied beyond the subset.
These are single-run timings on CPU. Prediction time is ColabFold's logged
per-model duration and can include compilation; wall time also includes process
startup, feature preparation and output generation. Do not treat these as a
controlled GPU throughput benchmark or evidence of a general speed trend.

Reproduce from the workspace root:

```bash
conda activate colabfold
python alphafold/benchmark_msa_depth.py \
  --msa alphafold/target_unfiltered.a3m \
  --output alphafold/msa_depth_benchmark
```

The driver reuses completed runs with matching inputs and configuration. It exits
with status 1 if any requested depth is unavailable or inference fails. To run
128 and 256 genuinely, supply a target-matched A3M alignment containing at least
256 distinct aligned sequences and select a new output directory.

`msa_depth_benchmark/metrics.csv` contains confidence percentages and timings.
Each `depth_*` directory contains the input A3M, predicted PDB, score JSON,
confidence plots, inference log and benchmark timing metadata.