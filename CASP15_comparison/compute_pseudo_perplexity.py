"""Compute ESM masked pseudo-perplexity with an existing LOCAL ESM-2 checkpoint.

Requires a separate environment with torch and fair-esm; does not download weights.
This is an ESM-2 sequence proxy, not an output of the ESMFold structure model.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("An existing local pretrained ESM-2 checkpoint is required")
    import torch
    import esm
    from Bio import SeqIO

    root = Path(__file__).resolve().parent
    output = args.output or root / "perplexity.csv"
    if output.exists() and output.read_text().count("\n") > 1:
        parser.error(f"{output} already contains measurements; choose another --output")
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(args.checkpoint))
    model = model.eval().to(args.device)
    converter = alphabet.get_batch_converter()
    rows = []
    for record in SeqIO.parse(root / "targets.fasta", "fasta"):
        sequence = str(record.seq)
        if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY") or not 1 <= len(sequence) <= 1022:
            raise ValueError(f"Invalid sequence or unsupported length: {record.id}")
        _, _, tokens = converter([(record.id, sequence)])
        tokens = tokens.to(args.device)
        log_probabilities = []
        with torch.inference_mode():
            for position in range(1, len(sequence) + 1):
                masked = tokens.clone()
                masked[0, position] = alphabet.mask_idx
                logits = model(masked, repr_layers=[], return_contacts=False)["logits"]
                log_probability = torch.log_softmax(logits[0, position].float(), dim=-1)
                log_probabilities.append(log_probability[tokens[0, position]].item())
        value = math.exp(-sum(log_probabilities) / len(log_probabilities))
        if not math.isfinite(value) or value < 1:
            raise ValueError(f"Invalid pseudo-perplexity for {record.id}")
        rows.append({
            "target": record.id, "value": value,
            "method": f"ESM2_masked_leave_one_out:{args.checkpoint.name}:sha256={digest.hexdigest()}",
        })
        print(f"{record.id}: masked pseudo-perplexity={value:.6f}", flush=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "value", "method"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
