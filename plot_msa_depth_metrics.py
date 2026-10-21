"""Plot completed ColabFold depth experiments without implying ESMFold results.

Run with the colabfold environment:
    python plot_msa_depth_metrics.py
Override locations with --results PATH and --output PATH.
Only ranked model 1 is analyzed; these are confidence scores, not measured
structural accuracy. Inference time is read from the ColabFold log and excludes
startup, compilation before prediction, and other wall-clock overhead.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "esm/esm/esmfold/v1/msa_depth_results"
METRICS = (
    ("mean_plddt", "Mean pLDDT", "pLDDT (0-100)", True),
    ("ptm", "Predicted TM score (pTM)", "pTM (0-1)", True),
    ("mean_pae_angstrom", "Mean predicted aligned error", "PAE (angstrom)", False),
    ("plddt_ge_70_percent", "Residues with pLDDT >= 70", "Residues (%)", True),
    ("plddt_ge_90_percent", "Residues with pLDDT >= 90", "Residues (%)", True),
    ("prediction_seconds", "Inference runtime", "Seconds", False),
)


def read_alignment(path):
    sequences = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            sequences.append("")
        elif line and not line.startswith("#"):
            if not sequences:
                raise ValueError(f"Sequence before header in {path}")
            sequences[-1] += line.strip()
    aligned = [re.sub(r"[a-z.]", "", sequence) for sequence in sequences]
    if not aligned or any(not sequence for sequence in aligned):
        raise ValueError(f"Empty alignment in {path}")
    if len({len(sequence) for sequence in aligned}) != 1:
        raise ValueError(f"Inconsistent aligned sequence lengths in {path}")
    return aligned


def load_results(results):
    rows, profiles, alignments = [], {}, []
    directories = sorted(results.glob("depth_*"), key=lambda p: int(p.name.split("_")[1]))
    if not directories:
        raise ValueError(f"No depth directories found in {results}")
    for directory in directories:
        depth = int(directory.name.split("_")[1])
        if depth <= 0:
            raise ValueError(f"Nonpositive depth in {directory}")
        score_files = list(directory.glob("*_scores_rank_001_*.json"))
        if len(score_files) != 1 or not (directory / "target.done.txt").exists():
            raise ValueError(f"Expected one completed rank-1 prediction in {directory}")
        config = json.loads((directory / "config.json").read_text())
        if config["model_type"] != "alphafold2_ptm" or config["max_seq"] != depth:
            raise ValueError(f"Unexpected model or mismatched configured depth in {directory}")
        scores = json.loads(score_files[0].read_text())
        plddt, pae = scores["plddt"], scores["pae"]
        if not plddt or len(pae) != len(plddt) or any(
            len(row) != len(plddt) for row in pae
        ):
            raise ValueError(f"Invalid confidence array dimensions in {score_files[0]}")
        values = [*plddt, scores["ptm"], *(value for row in pae for value in row)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite confidence values in {score_files[0]}")
        if not all(0 <= value <= 100 for value in plddt):
            raise ValueError(f"pLDDT outside [0, 100] in {score_files[0]}")
        if not 0 <= scores["ptm"] <= 1 or any(value < 0 for row in pae for value in row):
            raise ValueError(f"Invalid pTM or PAE in {score_files[0]}")
        aligned = read_alignment(directory / "target.a3m")
        if len(aligned[0]) != len(plddt):
            raise ValueError(f"Alignment and prediction lengths differ in {directory}")
        if alignments and aligned[0] != alignments[0][0]:
            raise ValueError("Cannot compare predictions for different query sequences")
        alignments.append(aligned)
        times = re.findall(
            r"alphafold2_ptm_model_\d+_seed_\d+ took ([\d.]+)s",
            (directory / "log.txt").read_text(),
        )
        if len(times) != 1:
            raise ValueError(f"Expected one unambiguous prediction runtime in {directory}")
        runtime = float(times[0])
        if not math.isfinite(runtime) or runtime <= 0:
            raise ValueError(f"Invalid runtime in {directory}")
        rows.append({
            "configured_max_seq": depth,
            "configured_max_extra_seq": config["max_extra_seq"],
            "input_records": len(aligned),
            "unique_aligned_sequences": len(set(aligned)),
            "residue_count": len(plddt),
            "mean_plddt": mean(plddt),
            "ptm": scores["ptm"],
            "mean_pae_angstrom": mean(value for row in pae for value in row),
            "plddt_ge_70_percent": 100 * sum(value >= 70 for value in plddt) / len(plddt),
            "plddt_ge_90_percent": 100 * sum(value >= 90 for value in plddt) / len(plddt),
            "prediction_seconds": runtime,
            "source_scores": str(score_files[0].resolve()),
        })
        profiles[depth] = plddt
    return rows, profiles, all(aligned == alignments[0] for aligned in alignments)


def draw_metric(ax, rows, metric):
    key, title, ylabel, higher = metric
    depths = [row["configured_max_seq"] for row in rows]
    values = [row[key] for row in rows]
    best = (max if higher else min)(values)
    best_indices = [
        index for index, value in enumerate(values)
        if math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-10)
    ]
    ax.plot(depths, values, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax.scatter(
        [depths[index] for index in best_indices],
        [values[index] for index in best_indices],
        marker="*", s=160, color="#16a34a", edgecolor="white", zorder=4,
        label="Best observed (ties included)",
    )
    for depth, value in zip(depths, values):
        ax.annotate(f"{value:.3f}", (depth, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    if key == "ptm":
        ax.set_ylim(max(0, best - 0.03), min(1, best + 0.03))
    else:
        ax.margins(y=0.28)
    ax.set_title(f"{title} | {'Higher' if higher else 'Lower'} is better", fontsize=12)
    ax.set_xlabel("Configured MSA limit (max_seq; not actual depth)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)


def write_report(output, results, rows, same_alignment):
    lines = [
        "MSA limit experiment: AlphaFold2 / ColabFold",
        f"Source: {results.resolve()}",
        "",
        "These are AlphaFold2 predictions, despite their location under the ESMFold tree.",
        "The separate alphafold/msa_depth_results folder contains prepared inputs only.",
        "No AlphaFold-versus-ESMFold comparison or experimental accuracy claim is possible.",
        "",
        "X axis: configured max_seq, NOT measured effective MSA depth.",
        "max_extra_seq also changes; effects cannot be attributed to max_seq alone.",
        f"All input alignments identical: {same_alignment}.",
        "Input availability (records include query and duplicates):",
    ]
    for row in rows:
        lines.append(
            f"  max_seq={row['configured_max_seq']}, "
            f"max_extra_seq={row['configured_max_extra_seq']}: "
            f"{row['input_records']} records, "
            f"{row['unique_aligned_sequences']} unique aligned sequences"
        )
    lines += ["", "Best observed value per metric (ties use stored score precision):"]
    for key, title, _, higher in METRICS:
        best = (max if higher else min)(row[key] for row in rows)
        depths = [
            str(row["configured_max_seq"]) for row in rows
            if math.isclose(row[key], best, rel_tol=1e-10, abs_tol=1e-10)
        ]
        lines.append(f"  {title}: {best:.6f}; max_seq = {', '.join(depths)}")
    lines += [
        "",
        "pLDDT and pTM are confidence estimates, not ground-truth accuracy metrics.",
        "Mean PAE averages the full matrix, including the diagonal.",
        "Runtime is the log's 'took ...s' prediction time, not end-to-end wall time.",
        "One rank-1 prediction per setting; no replicate uncertainty estimates.",
        "Small numerical differences do not establish statistically significant improvement.",
        "Limits exceeding available unique sequences do not add new sequence information.",
        "",
        "Reproduce: python plot_msa_depth_metrics.py --results PATH --output PATH",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / "msa_depth_plots")
    args = parser.parse_args()
    rows, profiles, same_alignment = load_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, metric in zip(axes.flat, METRICS):
        draw_metric(ax, rows, metric)
    fig.suptitle("AlphaFold2: confidence and runtime versus configured MSA limit",
                 fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.02,
        "Configured limits are not actual MSA depths; see summary.txt for input counts. "
        "max_extra_seq also varies.\n"
        "Single prediction per setting; confidence is not measured accuracy. "
        "No completed ESMFold results available.",
        ha="center", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.94), h_pad=3, w_pad=2)
    fig.savefig(args.output / "msa_depth_overview.png", dpi=200)
    fig.savefig(args.output / "msa_depth_overview.pdf")
    plt.close(fig)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9, 6))
        draw_metric(ax, rows, metric)
        fig.suptitle("AlphaFold2 / ColabFold", fontweight="bold")
        fig.text(0.5, 0.02, "Single run per setting; max_extra_seq also varies. "
                 "Confidence is not measured accuracy.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(args.output / f"{metric[0]}_vs_depth.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    for index, (depth, plddt) in enumerate(profiles.items()):
        ax.plot(range(1, len(plddt) + 1), plddt, label=f"max_seq={depth}",
                linewidth=1.5, linestyle=("-", "--", ":", "-.", "--")[index % 5])
    ax.axhline(70, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(90, color="gray", linestyle=":", alpha=0.5)
    ax.set(title="AlphaFold2: per-residue pLDDT (higher is better)",
           xlabel="Residue position (1-based)", ylabel="pLDDT", ylim=(0, 102))
    ax.legend(ncol=min(len(profiles), 5), loc="lower center", fontsize=9)
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.02, "Identical profiles overlap. Labels denote configured limits, "
             "not effective MSA depths.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.output / "per_residue_plddt.png", dpi=200)
    plt.close(fig)
    write_report(args.output, args.results, rows, same_alignment)
    print((args.output / "summary.txt").read_text())
    print(f"Saved plots and metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
