# CASP15: AlphaFold2 versus ESMFold

## Start here

- [plots/](plots/): regenerated per-target confidence comparisons, 3D CA-backbone
  overlays, accuracy panels, runtime comparison, and perplexity-versus-accuracy panels.
- [available_T1184_plots/](available_T1184_plots/): copies of the previously generated
  depth and combined-model figures, including PAE, pTM, runtime and per-residue confidence.
- [status.csv](status.csv): missing inputs and confidence-scale assumptions.
- [confidence_metrics.csv](confidence_metrics.csv), [accuracy_metrics.csv](accuracy_metrics.csv),
  and [recorded_runtimes.csv](recorded_runtimes.csv): underlying measurements.
- [targets.fasta](targets.fasta): user-confirmed target-to-sequence mapping.

Missing results appear as **unavailable**, never zero-valued accuracy.
Only T1184 predictions were present when the project was assembled.
T1187 and T1181 currently have input FASTAs and explicitly missing-result panels,
not completed predictions. No experimental references were present.
Target IDs and CASP15 FM classification are supplied by the user, not independently
certified by this workflow.

## Regenerate

From this folder, using the existing ColabFold environment:

```sh
/Users/sreebhargavibalija/miniforge3/envs/colabfold/bin/python analyze.py
```

Dependencies already present: NumPy, Matplotlib, Biopython.
The script reads [project.json](project.json), validates prediction sequences,
extracts data, makes figures and updates tables.
Original prediction files are read-only. Previously available T1184 figures
are copied into this folder without changing their provenance.
The shared plotting scripts one directory above are required.

## Add predictions for T1187 and T1181

Each target has `inputs/TARGET/query.fasta`. Run predictions on a suitable
GPU machine or use ColabFold, then supply the results locally.
**No new inference jobs or remote sequence submissions are performed by analyze.py.**
The current analysis environment lacks PyTorch and a configured ESMFold runner.

For standard AlphaFold2/ColabFold depth runs, use folders such as
`inputs/T1187/alphafold/depth_16/`, `depth_32/`, etc. Each needs:

- `target.a3m` (the exact MSA used, including query)
- `config.json` (`model_type=alphafold2_ptm`, `max_seq` matching folder depth,
  `max_extra_seq`)
- `target.done.txt`
- one `*_scores_rank_001_*.json`
- one `*_unrelaxed_rank_001_*.pdb`
- `log.txt` with exactly one model prediction `took ...s` entry.

Use the target filename stem `target` for completion and alignment files.
Model type must be AlphaFold2, not inferred from the containing folder name.
To use a different result location, edit [project.json](project.json).

For ESMFold, place a single-chain prediction at `inputs/TARGET/esmfold.pdb`.
The existing local ESMFold runner is `../esm/scripts/fold.py`; in a properly
provisioned ESMFold environment its interface is:

```sh
python ../esm/scripts/fold.py -i targets.fasta -o esmfold_predictions --chunk-size 128
```

This is a GPU/model-dependency-dependent prediction step, **not executed here**.
Map its `T1187.pdb` and `T1181.pdb` outputs in the configuration. Keep the original
logs and record timing scope and hardware in [timing.csv](timing.csv).
Do not assume a server response time equals GPU inference time.

## Confidence interpretation

ESMFold is MSA-independent: a horizontal line is one prediction, not multiple depth runs.
For T1184 only, the user approved multiplying PDB CA confidence values by 100.
This scale is **not documented in that PDB**, so conclusions remain conditional.
Other targets default to standard local ESMFold export scale 1 (0–100);
verify their actual provenance. One CA confidence value is used per residue,
not the atom-weighted ESMFold mean.
pLDDT and pTM are not experimental accuracy metrics.
The PDB alone supplies no ESMFold pTM, PAE, runtime, or perplexity.

The current T1184 MSA-limit experiment has identical input alignments
(64 records, 60 unique aligned sequences) at all configured limits.
`max_extra_seq` also varies. Limits 128 and 256 are not genuine depths with
128/256 unique sequences. Higher-depth results cannot establish a causal
depth improvement. One prediction per condition supplies no uncertainty estimate.

A separate experiment at `../alphafold/msa_depth_benchmark` was discovered.
It has different MSA settings and is intentionally not pooled with the existing
experiment. Analyze separately if needed, preserving its original logs and settings.

## Reference-based accuracy: TM-score, GDT-TS and RMSD

Obtain the correct experimental reference chain for each target and place it at
`inputs/TARGET/reference.pdb`, or update the configuration.
This initial workflow requires a single-chain, complete reference sequence
matching the supplied target, with one CA atom per residue and no alternate CA
records. Prediction and reference residue numbering must match (checked before
TMscore is invoked). Truncated chains, domain-only CASP references, mmCIF and residue
remapping need explicit preprocessing and coverage reporting before use.
Do not silently delete unmatched residues.

Provide the official **TMscore** executable (not TM-align/US-align, whose output
format differs):

```sh
python analyze.py --tmscore /absolute/path/to/TMscore
```

The tool is invoked as `TMscore prediction.pdb reference.pdb`.
TM-score is normalized to the reference (second structure); GDT-TS is reported
as a fraction 0–1; RMSD is in angstrom. Full matching target lengths are enforced.
The executable's output is retained beside the plots; parse failures stop the
workflow rather than yielding substitute scores.
Without the reference and executable, accuracy panels are explicitly unavailable.
3D overlays without a reference measure model agreement, not accuracy.

## Perplexity relationship

[perplexity.csv](perplexity.csv) accepts `target,value,method`, one measurement
per target and method, with value >=1. Use an explicit model/version/scoring
definition in `method`, and apply the same method to all targets.
Only measured perplexity and reference-based TM-score pairs are plotted.
Depth runs from the same target are not independent observations.
Three targets are insufficient for strong general claims; no significance or
regression is fabricated.

ESMFold does not directly expose a canonical sequence perplexity in its PDB.
ESM-2 **masked pseudo-perplexity** is a possible proxy, not “ESMFold perplexity”.
If using it, define
`exp(-mean_i log p(x_i | all other residues, position i masked))`,
state the exact pretrained model/checkpoint and masking policy, and keep it
separate from confidence/accuracy. No perplexity values have been invented.

The optional [compute_pseudo_perplexity.py](compute_pseudo_perplexity.py) automates
that exact leave-one-out definition for all three targets using an **existing,
trusted local ESM-2 checkpoint** in an environment with PyTorch and fair-esm:

```sh
python compute_pseudo_perplexity.py --checkpoint /path/to/esm2_checkpoint.pt --device cuda
python analyze.py --tmscore /path/to/TMscore
```

It does not download model weights, records the checkpoint SHA-256 in the method,
and refuses to overwrite existing measurements. It uses one masked forward pass
per residue (potentially slow). Some checkpoints need their companion local
contact-regression file as required by fair-esm's loader. This optional workflow
has not been executed here because PyTorch and checkpoint weights are unavailable.

## Validation

```sh
python -m unittest test_analysis.py
```

Tests cover rigid superposition, sequence mismatch rejection, the T1184
confidence-scale calculation, and TMscore text parsing. Parser tests use fixtures,
not a real TMscore run; real structural scoring remains unverified until the
executable and references are supplied.

## Remaining work

1. Generate/provide AlphaFold2 and ESMFold predictions for T1187 and T1181.
2. Obtain experimental target references and verify chain/domain mapping.
3. Supply TMscore for genuine structural evaluation.
4. Record comparable hardware and timing scope for both models.
5. Compute consistently defined sequence perplexity values.
6. Rerun analysis; replace missing-result panels with measured comparisons.

All existing plots are descriptive; they do not establish that one model has
higher experimental accuracy without reference-based evaluation.
