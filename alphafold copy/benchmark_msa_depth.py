import argparse
import csv
import json
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import time

from Bio import SeqIO


SEQUENCE = "MRSLKLIIINALTLIIGLGLMSGSVSQRSAEAARFETIPSLQVNKLGNYRGQYLTVLYAVGSRPFISTENSSISISQVKEARTVYISADSMSLPSVQVEKEGFRPSYNIVVFVVSPQANYSWVNADGSVPQGMTATGNRLSSLINAINKTDVDGFISAQGETGTLQINLVK"
FIELDS = (
    "requested_depth", "input_depth", "available_unique_sequences", "status",
    "mean_plddt", "ptm", "mean_pae_angstrom", "plddt_ge_70_percent",
    "plddt_ge_90_percent", "prediction_seconds", "wall_seconds",
)


def load_alignment(path):
    records, seen = [], set()
    with path.open() as handle:
        for record in SeqIO.parse(handle, "fasta-pearson"):
            sequence = str(record.seq)
            aligned = re.sub(r"[a-z.]", "", sequence)
            if len(aligned) != len(SEQUENCE):
                raise ValueError(f"Invalid aligned length for {record.id}: {len(aligned)}")
            if aligned not in seen:
                records.append((record.description, sequence))
                seen.add(aligned)
    if not records or re.sub(r"[a-z.]", "", records[0][1]) != SEQUENCE:
        raise ValueError("The first MSA record must match the target sequence")
    return records


def write_metrics(output, rows):
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    records = load_alignment(args.msa)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(args.colabfold)
    if not args.prepare_only and executable is None:
        raise ValueError(f"ColabFold executable not found: {args.colabfold}")
    rows = []
    for depth in args.depths:
        row = dict(requested_depth=depth, input_depth="",
                   available_unique_sequences=len(records), status="insufficient_sequences")
        rows.append(row)
        if depth > len(records):
            print(f"Depth {depth}: unavailable; only {len(records)} unique sequences", flush=True)
            write_metrics(output, rows)
            continue
        directory = output / f"depth_{depth}"
        directory.mkdir(exist_ok=True)
        alignment = directory / "target.a3m"
        content = "".join(f">{header}\n{sequence}\n" for header, sequence in records[:depth])
        if alignment.exists() and alignment.read_text() != content:
            raise ValueError(f"Different alignment in {directory}; choose a new output directory")
        alignment.write_text(content)
        row.update(input_depth=depth, status="prepared")
        write_metrics(output, rows)
        if args.prepare_only:
            print(f"Depth {depth}: prepared exactly {depth} unique sequences", flush=True)
            continue
        command = [
            executable, str(alignment), str(directory),
            "--model-type", "alphafold2_ptm", "--num-models", "1",
            "--model-order", "1", "--num-seeds", "1", "--random-seed", "0",
            "--num-recycle", "3", "--max-msa", f"{depth}:1",
        ]
        metadata = directory / "benchmark.json"
        previous = json.loads(metadata.read_text()) if metadata.exists() else None
        if previous and previous["command"] != command:
            raise ValueError(f"Different inference configuration in {directory}")
        finished = directory / "target.done.txt"
        if not (finished.exists() and previous and previous["returncode"] == 0):
            print(f"Depth {depth}: starting AlphaFold2 inference", flush=True)
            row["status"] = "running"
            write_metrics(output, rows)
            start = time.perf_counter()
            with (directory / "run.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            previous = dict(command=command, returncode=result.returncode,
                            wall_seconds=time.perf_counter() - start)
            metadata.write_text(json.dumps(previous, indent=2) + "\n")
        row["wall_seconds"] = round(previous["wall_seconds"], 3)
        score_files = list(directory.glob("target_scores_rank_001_*.json"))
        if previous["returncode"] or not finished.exists() or len(score_files) != 1:
            row["status"] = "failed"
            print(f"Depth {depth}: failed; see {directory / 'run.log'}", flush=True)
            write_metrics(output, rows)
            continue
        scores = json.loads(score_files[0].read_text())
        plddt = scores["plddt"]
        elapsed = re.search(r"took ([\d.]+)s", (directory / "run.log").read_text())
        row.update(
            status="completed", mean_plddt=round(mean(plddt), 3), ptm=scores["ptm"],
            mean_pae_angstrom=round(mean(value for values in scores["pae"] for value in values), 3),
            plddt_ge_70_percent=round(100 * sum(value >= 70 for value in plddt) / len(plddt), 3),
            plddt_ge_90_percent=round(100 * sum(value >= 90 for value in plddt) / len(plddt), 3),
            prediction_seconds=float(elapsed[1]) if elapsed else "",
        )
        write_metrics(output, rows)
        print(f"Depth {depth}: pLDDT={row['mean_plddt']}, pTM={row['ptm']}, "
              f"PAE={row['mean_pae_angstrom']} A", flush=True)
    print(f"Metrics: {output / 'metrics.csv'}", flush=True)
    return 0 if all(row["status"] in ("completed", "prepared") for row in rows) else 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark exact MSA depths including the query.")
    parser.add_argument("--msa", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "msa_depth_results")
    parser.add_argument("--depths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--colabfold", default="colabfold_batch")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if any(depth < 1 for depth in args.depths) or len(args.depths) != len(set(args.depths)):
        parser.error("Depths must be distinct positive integers")
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())