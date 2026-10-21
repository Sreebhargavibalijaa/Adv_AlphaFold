"""Build CASP15 plots from verified inputs; never substitute confidence for accuracy."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import seq1
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from plot_msa_depth_metrics import load_results
from plot_model_comparison import METRICS, read_esmfold


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ca_structure(path, query):
    lines = path.read_text().splitlines()
    if sum(line.startswith("MODEL ") for line in lines) > 1:
        raise ValueError(f"Multiple structure models in {path}")
    atoms = [line for line in lines
             if line.startswith("ATOM  ") and line[12:16].strip() == "CA"]
    if len({line[21:27] for line in atoms}) != len(atoms):
        raise ValueError(f"Duplicate CA residues in {path}")
    sequence = "".join(seq1(line[17:20]) for line in atoms)
    if sequence != query:
        raise ValueError(f"Structure sequence does not exactly match target in {path}")
    xyz = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    for line in atoms])
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite coordinates in {path}")
    return xyz


def superpose(mobile, reference):
    centered = mobile - mobile.mean(axis=0)
    ref = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(centered.T @ ref)
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    return centered @ u @ correction @ vt


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def unavailable(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, wrap=True)
    ax.set_axis_off()


def confidence_figure(target, af, esm, output):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (key, title, ylabel) in zip(axes, METRICS):
        if not af and esm is None:
            unavailable(ax, "Missing AlphaFold2 and ESMFold predictions")
            continue
        if af:
            depths = [row["configured_max_seq"] for row in af]
            ax.plot(depths, [row[key] for row in af], "o-", label="AlphaFold2",
                    color="#2563eb")
            ax.set_xscale("log", base=2)
            ax.set_xticks(depths)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if esm is not None:
            ax.axhline(esm[key], linestyle="--", color="#ea580c",
                       label=f"ESMFold baseline: {esm[key]:.2f}")
        ax.set(title=f"{title} (higher is better)", ylabel=ylabel,
               xlabel="AlphaFold configured MSA limit", ylim=(-4, 105))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"{target}: AlphaFold2 vs ESMFold confidence", fontsize=17)
    fig.text(0.5, 0.025,
             "Confidence is not accuracy. ESMFold uses no MSA. See status.csv for "
             "missing inputs and per-target scale assumptions.\n"
             "Configured MSA limits are not measured depths; max_extra_seq may also vary.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    save(fig, output / "confidence_comparison")


def structures_figure(target, structures, output):
    fig = plt.figure(figsize=(12, 5))
    if not structures:
        unavailable(fig.add_subplot(111), f"{target}: missing prediction structures")
    else:
        reference = structures.get("Reference", next(iter(structures.values())))
        ax = fig.add_subplot(121, projection="3d")
        for label, xyz in structures.items():
            aligned = superpose(xyz, reference)
            ax.plot(*aligned.T, label=label, linewidth=1.1)
        all_xyz = np.concatenate([superpose(xyz, reference) for xyz in structures.values()])
        low, high = all_xyz.min(axis=0), all_xyz.max(axis=0)
        center, radius = (low + high) / 2, max(float((high - low).max()) / 2, 1)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set(title=f"{target}: CA-backbone overlay", xlabel="X (angstrom)",
               ylabel="Y (angstrom)", zlabel="Z (angstrom)")
        ax.legend(fontsize=8)
        ax.set_box_aspect((1, 1, 1))
        ax2 = fig.add_subplot(122)
        for label, xyz in structures.items():
            aligned = superpose(xyz, reference)
            ax2.plot(np.arange(1, len(xyz) + 1),
                     np.linalg.norm(aligned - (reference - reference.mean(axis=0)), axis=1),
                     label=label)
        ax2.set(title="CA distance after global least-squares fit",
                xlabel="Residue", ylabel="Distance (angstrom)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)
        fig.text(0.5, 0.02,
                 "Reference: experimental structure if provided; otherwise first prediction "
                 "(model agreement only, NOT accuracy).",
                 ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, output / "structure_overlay")


def score_structure(executable, prediction, reference, logfile):
    result = subprocess.run([executable, str(prediction), str(reference)],
                            capture_output=True, text=True, check=True, timeout=300)
    logfile.write_text(result.stdout + result.stderr)
    patterns = {
        "tm_score": r"TM-score\s*=\s*([\d.]+)",
        "gdt_ts": r"GDT-TS-score\s*=\s*([\d.]+)",
        "rmsd": r"RMSD of\s+(?:the\s+)?common residues\s*=\s*([\d.]+)",
    }
    scores = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, result.stdout)
        if len(matches) != 1:
            raise ValueError(f"Cannot unambiguously parse {key}; inspect {logfile}")
        scores[key] = float(matches[0])
    if not 0 <= scores["tm_score"] <= 1 or not 0 <= scores["gdt_ts"] <= 1:
        raise ValueError(f"Invalid normalized structural score in {logfile}")
    if not math.isfinite(scores["rmsd"]) or scores["rmsd"] < 0:
        raise ValueError(f"Invalid RMSD in {logfile}")
    return scores


def accuracy_figure(target, evaluations, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes, ("tm_score", "gdt_ts", "rmsd"),
                              ("TM-score (higher is better)", "GDT-TS (higher is better)",
                               "RMSD, angstrom (lower is better)")):
        if not evaluations:
            unavailable(ax, "Accuracy unavailable:\nreference structure and TMscore required")
            continue
        ax.bar(range(len(evaluations)), [row[key] for row in evaluations])
        ax.set_xticks(range(len(evaluations)),
                      [f"{row['model']}\n{row['depth']}" for row in evaluations],
                      rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
    fig.suptitle(f"{target}: accuracy against experimental reference")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output / "accuracy_comparison")


def plot_timing(timings, output):
    fig, ax = plt.subplots(figsize=(13, 6))
    if not timings:
        unavailable(ax, "No recorded runtimes")
    else:
        labels = [f"{row['target']} {row['model']} {row['depth']}\n"
                  f"{row['hardware']} | {row['timing_scope']}" for row in timings]
        ax.bar(range(len(timings)), [row["seconds"] for row in timings])
        ax.set_xticks(range(len(timings)), labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Seconds (lower is faster)")
        ax.set_title("Recorded runtimes: compare only matching hardware and timing scope")
    fig.tight_layout()
    save(fig, output / "runtime_comparison")


def plot_perplexity(perplexities, evaluations, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    joined = [(p, e) for p in perplexities for e in evaluations if p["target"] == e["target"]]
    methods = sorted({p["method"] for p, _ in joined})
    for ax, model in zip(axes, ("AlphaFold2", "ESMFold")):
        points = [(p, e) for p, e in joined if e["model"] == model]
        if not points:
            unavailable(ax, f"{model}: missing perplexity and/or reference-based accuracy")
            continue
        for method in methods:
            subset = [(p, e) for p, e in points if p["method"] == method]
            ax.scatter([p["value"] for p, _ in subset],
                       [e["tm_score"] for _, e in subset], label=method)
            for p, e in subset:
                ax.annotate(f"{p['target']} {e['depth']}", (p["value"], e["tm_score"]),
                            fontsize=8)
        ax.set(title=model, xlabel="Sequence perplexity (method in legend)",
               ylabel="Reference-based TM-score", ylim=(0, 1.05))
        ax.legend(fontsize=8)
    fig.suptitle("Perplexity vs accuracy: depth settings are not independent targets")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, output / "perplexity_vs_accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmscore", help="Path to the official TMscore executable")
    args = parser.parse_args()
    executable = shutil.which(args.tmscore or "TMscore")
    if args.tmscore and executable is None:
        parser.error(f"TMscore executable not found: {args.tmscore}")
    config = json.loads((ROOT / "project.json").read_text())
    targets = {record.id: str(record.seq) for record in SeqIO.parse(ROOT / "targets.fasta", "fasta")}
    if set(targets) != set(config):
        raise ValueError("Target FASTA and project configuration IDs differ")
    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    status, confidence, evaluations, timings = [], [], [], []
    for target, query in targets.items():
        directory = output / target
        directory.mkdir(exist_ok=True)
        inputs = ROOT / "inputs" / target
        inputs.mkdir(parents=True, exist_ok=True)
        (inputs / "query.fasta").write_text(f">{target}\n{query}\n")
        if not query or set(query) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"Invalid amino acid sequence for {target}")
        paths = {key: (ROOT / config[target][key]).resolve()
                 for key in ("alphafold_results", "esmfold_pdb", "reference_pdb")}
        af, esm, structures, predictions = [], None, {}, []
        af_path = paths["alphafold_results"]
        if af_path.exists() and list(af_path.glob("depth_*")):
            af, _, _ = load_results(af_path)
            for row in af:
                depth = row["configured_max_seq"]
                pdbs = list((af_path / f"depth_{depth}").glob("*_unrelaxed_rank_001_*.pdb"))
                if len(pdbs) != 1:
                    raise ValueError(f"Expected one ranked PDB at {af_path}/depth_{depth}")
                xyz = ca_structure(pdbs[0], query)
                predictions.append(("AlphaFold2", depth, pdbs[0]))
                structures[f"AlphaFold2 limit={depth}"] = xyz
                confidence.append({"target": target, "model": "AlphaFold2",
                                   "depth": depth, **{key: row[key] for key, _, _ in METRICS}})
                timings.append({"target": target, "model": "AlphaFold2", "depth": depth,
                                "seconds": row["prediction_seconds"],
                                "hardware": (
                                    "CPU (run log)" if "no GPU detected" in
                                    (af_path / f"depth_{depth}" / "log.txt").read_text()
                                    else "Unknown (verify run hardware)"
                                ),
                                "timing_scope": "logged prediction; not end-to-end"})
        if paths["esmfold_pdb"].exists():
            esm = read_esmfold(paths["esmfold_pdb"], query,
                               config[target]["esmfold_confidence_scale"])
            structures["ESMFold"] = ca_structure(paths["esmfold_pdb"], query)
            predictions.append(("ESMFold", "", paths["esmfold_pdb"]))
            confidence.append({"target": target, "model": "ESMFold", "depth": "", **esm})
        native = paths["reference_pdb"]
        if native.exists():
            structures["Reference"] = ca_structure(native, query)
            if executable:
                for model, depth, prediction in predictions:
                    def residue_numbers(path):
                        return [line[22:27] for line in path.read_text().splitlines()
                                if line.startswith("ATOM  ") and line[12:16].strip() == "CA"]

                    if residue_numbers(prediction) != residue_numbers(native):
                        raise ValueError(
                            f"TMscore requires matching residue numbering: {prediction} vs {native}. "
                            "Explicitly remap the reference before scoring."
                        )
                    scores = score_structure(
                        executable, prediction, native,
                        directory / f"TMscore_{model}_{depth or 'baseline'}.txt")
                    evaluations.append({"target": target, "model": model, "depth": depth, **scores})
        issues = []
        if not af:
            issues.append("Missing AlphaFold2 predictions")
        if esm is None:
            issues.append("Missing ESMFold prediction")
        if not native.exists():
            issues.append("Missing experimental reference")
        if executable is None:
            issues.append("TMscore executable unavailable")
        status.append({"target": target, "residues": len(query),
                       "alphafold_runs": len(af), "esmfold_available": esm is not None,
                       "reference_available": native.exists(),
                       "esmfold_scale": config[target]["esmfold_confidence_scale"],
                       "scale_note": config[target]["esmfold_scale_note"],
                       "status": "; ".join(issues)})
        confidence_figure(target, af, esm, directory)
        structures_figure(target, structures, directory)
        accuracy_figure(target, [e for e in evaluations if e["target"] == target], directory)

    with (ROOT / "timing.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["target"] not in targets or row["model"] not in ("AlphaFold2", "ESMFold"):
                raise ValueError(f"Unknown target/model in timing.csv: {row}")
            row["seconds"] = float(row["seconds"])
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError("Runtime must be finite and positive")
            if not row["hardware"].strip() or not row["timing_scope"].strip():
                raise ValueError("Runtime needs hardware and timing_scope")
            timings.append(row)
    with (ROOT / "perplexity.csv").open() as handle:
        perplexities = list(csv.DictReader(handle))
    seen = set()
    for row in perplexities:
        row["value"] = float(row["value"])
        key = (row["target"], row["method"])
        if (row["target"] not in targets or not row["method"].strip()
                or not math.isfinite(row["value"]) or row["value"] < 1 or key in seen):
            raise ValueError(f"Invalid or duplicate perplexity measurement: {row}")
        seen.add(key)
    for row in status:
        missing = []
        if not any(p["target"] == row["target"] for p in perplexities):
            missing.append("Missing sequence perplexity")
        if not any(t["target"] == row["target"] and t["model"] == "ESMFold" for t in timings):
            missing.append("Missing ESMFold runtime")
        row["status"] = "; ".join(filter(None, [row["status"], *missing])) or "Inputs available"
    plot_timing(timings, output)
    plot_perplexity(perplexities, evaluations, output)
    write_csv(ROOT / "status.csv", status, list(status[0]))
    write_csv(ROOT / "confidence_metrics.csv", confidence,
              ["target", "model", "depth", *(metric[0] for metric in METRICS)])
    write_csv(ROOT / "accuracy_metrics.csv", evaluations,
              ["target", "model", "depth", "tm_score", "gdt_ts", "rmsd"])
    write_csv(ROOT / "recorded_runtimes.csv", timings,
              ["target", "model", "depth", "seconds", "hardware", "timing_scope"])
    archive = ROOT / "available_T1184_plots"
    archive.mkdir(exist_ok=True)
    for path in (ROOT.parent / "msa_depth_plots").iterdir():
        if path.is_file() and path.suffix in (".png", ".pdf", ".csv", ".txt"):
            shutil.copy2(path, archive / path.name)
    for row in status:
        print(f"{row['target']} ({row['residues']} residues): {row['status']}")
    print(f"Project plots saved in {output}")


if __name__ == "__main__":
    main()
