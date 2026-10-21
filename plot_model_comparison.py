"""Overlay AlphaFold2 depth runs and a single MSA-independent ESMFold baseline.

Run: python plot_model_comparison.py --esmfold-scale 100
The scale must be explicitly supplied because the PDB does not document it.
Only CA-atom confidence is used, giving one value per residue for both models.
ESMFold pTM, PAE, and runtime cannot be recovered from this PDB.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from Bio.SeqUtils import seq1

from plot_msa_depth_metrics import DEFAULT_RESULTS, ROOT, load_results, plt, read_alignment
from matplotlib.ticker import ScalarFormatter


METRICS = (
    ("mean_plddt", "Mean pLDDT", "pLDDT (0-100)"),
    ("plddt_ge_70_percent", "Residues with pLDDT >= 70", "Residues (%)"),
    ("plddt_ge_90_percent", "Residues with pLDDT >= 90", "Residues (%)"),
)


def read_esmfold(path, query, scale):
    text = path.read_text()
    if "ESMFOLD" not in text.upper():
        raise ValueError(f"No ESMFold identification in {path}")
    if sum(line.startswith("MODEL ") for line in text.splitlines()) > 1:
        raise ValueError(f"Multiple models in {path}")
    atoms = [
        line for line in text.splitlines()
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA"
    ]
    if not atoms or len({line[21:27] for line in atoms}) != len(atoms):
        raise ValueError(f"Missing or duplicate CA residues in {path}")
    sequence = "".join(seq1(line[17:20]) for line in atoms)
    if sequence != query:
        raise ValueError("ESMFold and AlphaFold query sequences do not match")
    values = [float(line[60:66]) * scale for line in atoms]
    if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
        raise ValueError("Scaled ESMFold confidence is outside [0, 100]")
    return {
        "mean_plddt": mean(values),
        "plddt_ge_70_percent": 100 * sum(value >= 70 for value in values) / len(values),
        "plddt_ge_90_percent": 100 * sum(value >= 90 for value in values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--esmfold-pdb", type=Path, default=ROOT / "sequence_esmfold_prediction.pdb")
    parser.add_argument("--esmfold-scale", type=float, required=True, choices=(1, 100))
    parser.add_argument("--output", type=Path, default=ROOT / "msa_depth_plots")
    args = parser.parse_args()
    rows, _, _ = load_results(args.results)
    first_depth = rows[0]["configured_max_seq"]
    query = read_alignment(args.results / f"depth_{first_depth}" / "target.a3m")[0]
    baseline = read_esmfold(args.esmfold_pdb, query, args.esmfold_scale)
    args.output.mkdir(parents=True, exist_ok=True)
    depths = [row["configured_max_seq"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (key, title, ylabel) in zip(axes, METRICS):
        values = [row[key] for row in rows]
        ax.plot(depths, values, "o-", color="#2563eb", linewidth=2.5,
                label="AlphaFold2 (MSA-limit runs)")
        ax.axhline(baseline[key], color="#ea580c", linestyle="--", linewidth=2.5,
                   label=f"ESMFold baseline ({baseline[key]:.2f})")
        for depth, value in zip(depths, values):
            ax.annotate(f"{value:.2f}", (depth, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-4, 105)
        ax.set_title(f"{title}\nHigher is better")
        ax.set_xlabel("AlphaFold configured MSA limit (max_seq)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="center right", fontsize=9)
    fig.suptitle("AlphaFold2 vs ESMFold: confidence comparison",
                 fontsize=19, fontweight="bold")
    fig.text(
        0.5, 0.025,
        f"ESMFold: one prediction, no MSA; CA B-factor x{args.esmfold_scale:g} "
        "(user-approved scale assumption, not documented in PDB).\n"
        "AlphaFold: configured limits, not actual depths; 60 unique input sequences; "
        "max_extra_seq also varies.\n"
        "Confidence is not measured accuracy. ESMFold pTM, PAE and runtime are unavailable.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94), w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"alphafold_vs_esmfold.{suffix}", dpi=200)
    plt.close(fig)

    fields = ["model", "configured_max_seq", *(metric[0] for metric in METRICS),
              "confidence_scale", "source"]
    with (args.output / "model_comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": "AlphaFold2",
                "configured_max_seq": row["configured_max_seq"],
                **{key: row[key] for key, _, _ in METRICS},
                "confidence_scale": 1,
                "source": row["source_scores"],
            })
        writer.writerow({
            "model": "ESMFold",
            "configured_max_seq": "",
            **baseline,
            "confidence_scale": args.esmfold_scale,
            "source": str(args.esmfold_pdb.resolve()),
        })
    lines = [
        "AlphaFold2 versus ESMFold confidence comparison",
        f"Identical query sequences verified: {len(query)} residues.",
        f"ESMFold source: {args.esmfold_pdb.resolve()}",
        f"ESMFold CA B-factor multiplied by {args.esmfold_scale:g}.",
        "Scale is a user-approved assumption; it is not documented in the PDB.",
        "CA-atom confidence gives one score per residue, not the ESMFold all-atom mean.",
        "ESMFold uses no MSA: the horizontal line is one baseline, not repeated depth runs.",
        "AlphaFold x-axis is configured max_seq, not actual/effective MSA depth.",
        "All supplied AlphaFold inputs have 64 records / 60 unique aligned sequences.",
        "max_extra_seq varies too; this is not an isolated max_seq experiment.",
        "ESMFold pTM, PAE, and runtime are not available in the PDB and are not plotted.",
        "Confidence does not establish accuracy or statistical significance.",
        "",
    ]
    for key, title, _ in METRICS:
        lines.append(f"{title}: ESMFold baseline = {baseline[key]:.6f}")
        for row in rows:
            lines.append(f"  AlphaFold2 max_seq={row['configured_max_seq']}: {row[key]:.6f}")
    (args.output / "model_comparison_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved combined figure to {args.output / 'alphafold_vs_esmfold.png'}")


if __name__ == "__main__":
    main()
