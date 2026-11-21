import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from seqtrust import SeqTrust
import seqtrust.tools.blat as blat_tools
import seqtrust.tools.motifs as motif_tools


def make_trust() -> SeqTrust:
    return SeqTrust(
        pd.DataFrame(
            {
                "Sequence": ["ACGT", "GGGG", "ATATAT"],
                "source": ["natural", "generated", "generated"],
            }
        )
    )


class SeqTrustTests(unittest.TestCase):
    def test_public_api_contract(self):
        trust = make_trust()
        expected = {
            "add_GC_content",
            "add_complexity_score",
            "add_max_repeat_length",
            "add_max_tandem_repeat_length",
            "add_max_dispersed_repeat_count",
            "add_kmer_diversity",
            "add_seq2function_embedding",
            "add_DNABERT2_embedding",
            "add_kmer_embedding",
            "add_seq2function_prediction",
            "add_seq2function_attribution",
            "add_seqlets",
            "add_seqlet_features",
            "add_motif_features",
            "add_motif_diversity",
            "add_motif_recovery",
            "blat_match",
        }
        self.assertTrue(expected.issubset(set(dir(trust.cal))))
        for removed_namespace in ("within", "between", "pipe", "train"):
            self.assertFalse(hasattr(trust, removed_namespace))

    def test_basic_metrics_apply_to_all_rows(self):
        trust = make_trust()
        trust.cal.add_GC_content()
        trust.cal.add_complexity_score()
        trust.cal.add_max_repeat_length()
        trust.cal.add_max_tandem_repeat_length(min_unit_size=2)
        trust.cal.add_max_dispersed_repeat_count(unit_size=2)
        trust.cal.add_kmer_diversity(kmer_size=2)
        trust.cal.add_kmer_embedding(kmer_size=2)

        np.testing.assert_allclose(trust.df["GC_content"], [0.5, 1.0, 0.0])
        self.assertEqual(trust.df["max_repeat_length"].tolist(), [1, 4, 1])
        self.assertTrue(all(len(vector) == 16 for vector in trust.df["kmer_embedding"]))
        self.assertEqual(len(trust.history), 7)

    def test_save_and_load(self):
        trust = make_trust()
        trust.cal.add_GC_content()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "state.pkl"
            trust.save(output)
            restored = SeqTrust.load(output)

        pd.testing.assert_frame_equal(restored.df, trust.df)
        self.assertEqual(restored.info["result_keys"], [])

    def test_motif_scan_uses_all_rows_and_blat_wrapper(self):
        trust = SeqTrust(
            pd.DataFrame(
                {
                    "Sequence": ["AAAA", "CCCC", "GGGG"],
                    "source": ["natural", "generated", "natural"],
                },
                index=[10, 20, 30],
            )
        )
        scanned_lengths = []

        def fake_scan(frame, **kwargs):
            scanned_lengths.append(len(frame))
            output_col = kwargs["feature_col"]
            result = frame.reset_index(drop=True).copy()
            result[output_col] = [np.array([1, 0], dtype=np.float32)] * len(result)
            scan = pd.DataFrame(
                {
                    "ID": ["0", "2"],
                    "Matrix_id": ["M1", "M1"],
                    "strand": ["+", "+"],
                    "start": [0, 0],
                    "end": [2, 2],
                    "score": [3.0, 3.0],
                    "pval": [1e-6, 1e-6],
                    "qval": [1e-5, 1e-5],
                }
            )
            return result, ["M1", "M2"], scan

        with patch.object(motif_tools, "add_motif_features", side_effect=fake_scan):
            trust.cal.add_motif_features()

        self.assertEqual(scanned_lengths, [3])
        self.assertEqual(trust.results["motif_scan"]["ID"].tolist(), ["0", "2"])
        trust.cal.add_motif_diversity()
        trust.cal.add_motif_recovery(["M1"])
        self.assertEqual(trust.df["motif_recovery_ratio"].tolist(), [1.0, 1.0, 1.0])

        summary = pd.DataFrame({"ok": [1]})
        with patch.object(blat_tools, "blat_group_match", return_value=summary) as mocked:
            result = trust.cal.blat_match(
                category_col="source",
                dataset_class="natural",
            )
        self.assertIs(result, summary)
        self.assertIsNone(mocked.call_args.kwargs["query_group"])
        self.assertEqual(mocked.call_args.kwargs["group_cols"], ["source"])


if __name__ == "__main__":
    unittest.main()
