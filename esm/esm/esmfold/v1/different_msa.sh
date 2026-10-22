#!/usr/bin/env bash
set -euo pipefail

if ! command -v colabfold_batch >/dev/null 2>&1; then
  printf '%s\n' \
    'Error: colabfold_batch is not on PATH.' \
    'Activate your ColabFold environment before running this script.' \
    'If ColabFold is not installed, follow the installation instructions:' \
    'https://github.com/sokrypton/ColabFold#installation' >&2
  if [[ "$(uname -s)" == 'Darwin' ]]; then
    printf '%s\n' \
      'This is macOS. Use a configured Linux/NVIDIA GPU environment or Google Colab for the GPU benchmark.' \
      'Google Colab: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb' >&2
  fi
  exit 127
fi

mkdir -p msa_depth_results

printf '>target\n%s\n' \
  'MRSLKLIIINALTLIIGLGLMSGSVSQRSAEAARFETIPSLQVNKLGNYRGQYLTVLYAVGSRPFISTENSSISISQVKEARTVYISADSMSLPSVQVEKEGFRPSYNIVVFVVSPQANYSWVNADGSVPQGMTATGNRLSSLINAINKTDVDGFISAQGETGTLQINLVK' \
  > msa_depth_results/input.fasta

# Reuse the same alignment for every experiment.
colabfold_batch \
  msa_depth_results/input.fasta \
  msa_depth_results/alignment \
  --msa-only

for depth in 16 32 64 128 256; do
  output="msa_depth_results/depth_${depth}"
  mkdir -p "$output"

  /usr/bin/time -p colabfold_batch \
    msa_depth_results/alignment/target.a3m \
    "$output" \
    --model-type alphafold2_ptm \
    --num-models 1 \
    --num-seeds 1 \
    --random-seed 0 \
    --num-recycle 3 \
    --max-msa "${depth}:$((depth * 2))" \
    > "$output/run.log" 2> "$output/timing.txt"
done