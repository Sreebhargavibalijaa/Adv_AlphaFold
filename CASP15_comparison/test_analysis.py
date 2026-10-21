import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import analyze
from plot_model_comparison import read_esmfold


class AnalysisTests(unittest.TestCase):
    def test_superposition_removes_rigid_rotation_and_translation(self):
        reference = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        mobile = reference @ rotation + np.array([4., 7., 9.])
        np.testing.assert_allclose(analyze.superpose(mobile, reference),
                                   reference - reference.mean(axis=0), atol=1e-12)

    def test_esmfold_sequence_and_scaling(self):
        with (analyze.ROOT / "targets.fasta").open() as handle:
            query = str(next(analyze.SeqIO.parse(handle, "fasta")).seq)
        pdb = analyze.ROOT.parent / "sequence_esmfold_prediction.pdb"
        scores = read_esmfold(pdb, query, 100)
        self.assertAlmostEqual(scores["mean_plddt"], 35.61403508771929)
        self.assertEqual(scores["plddt_ge_70_percent"], 0)
        with self.assertRaisesRegex(ValueError, "do not match"):
            read_esmfold(pdb, "A" + query[1:], 100)

    def test_reference_sequence_mismatch_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            analyze.ca_structure(analyze.ROOT.parent / "sequence_esmfold_prediction.pdb", "AAA")

    def test_tmscore_output_parsing_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "score.txt"
            completed = SimpleNamespace(
                stdout="TM-score = 0.75\nGDT-TS-score= 0.65\n"
                       "RMSD of  the common residues= 2.50\n",
                stderr="",
            )
            with patch.object(analyze.subprocess, "run", return_value=completed):
                scores = analyze.score_structure("TMscore", Path("pred"), Path("ref"), log)
            self.assertEqual(scores, {"tm_score": 0.75, "gdt_ts": 0.65, "rmsd": 2.5})
            with patch.object(analyze.subprocess, "run",
                              return_value=SimpleNamespace(stdout="incomplete", stderr="")):
                with self.assertRaisesRegex(ValueError, "Cannot unambiguously parse"):
                    analyze.score_structure("TMscore", Path("pred"), Path("ref"), log)


if __name__ == "__main__":
    unittest.main()
