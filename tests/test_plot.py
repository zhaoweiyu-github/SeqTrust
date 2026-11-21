import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust import SeqTrust


def make_plot_trust() -> SeqTrust:
    rng = np.random.default_rng(42)
    rows = []
    tools = ["Tool A", "Tool B", "Tool C", "Tool D"]
    datasets = ["Dataset 1", "Dataset 2", "Dataset 3"]
    for dataset_index, dataset in enumerate(datasets):
        for tool_index, tool in enumerate(tools):
            for replicate in range(3):
                center = 0.42 + 0.09 * tool_index + 0.035 * dataset_index
                vector = rng.normal(
                    loc=[tool_index, dataset_index, replicate / 3, tool_index / 2],
                    scale=0.12,
                )
                rows.append(
                    {
                        "Sequence": "ACGT" * (replicate + 1),
                        "tool": tool,
                        "dataset": dataset,
                        "score": center + rng.normal(0, 0.025),
                        "embedding": vector,
                        "label": f"Class {tool_index % 3 + 1}",
                    }
                )
    return SeqTrust(pd.DataFrame(rows))


class RankingTests(unittest.TestCase):
    def test_rank_tools_global_and_by_category(self):
        trust = make_plot_trust()
        global_ranks = trust.rank_tools(
            value_col="score",
            tool_col="tool",
            aggregation="topn",
            top_n=2,
        )
        self.assertEqual(global_ranks.iloc[0]["tool"], "Tool D")
        self.assertEqual(global_ranks["rank"].tolist(), [1, 2, 3, 4])
        self.assertTrue((global_ranks["aggregation"] == "topn").all())

        category_ranks = trust.rank_tools(
            value_col="score",
            tool_col="tool",
            aggregation="mean",
            category_col="dataset",
        )
        self.assertEqual(len(category_ranks), 12)
        for _, panel in category_ranks.groupby("dataset"):
            self.assertEqual(sorted(panel["rank"].tolist()), [1, 2, 3, 4])


@unittest.skipUnless(
    __import__("importlib").util.find_spec("matplotlib") is not None,
    "plot optional dependencies are not installed",
)
class PlotTests(unittest.TestCase):
    def assert_pdf(self, path: Path) -> None:
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1_000)
        self.assertEqual(path.read_bytes()[:4], b"%PDF")

    def test_all_plot_types_write_pdf(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trust = make_plot_trust()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            figure, result = trust.plot.performance_matrix(
                tool_col="tool",
                category_col="dataset",
                value_col="score",
                aggregation="topn",
                top_n=2,
                tools_on="rows",
                output_path=output_dir / "performance_rows.pdf",
            )
            self.assertEqual(result["matrix"].shape, (4, 3))
            plt.close(figure)

            figure, _ = trust.plot.performance_matrix(
                tool_col="tool",
                category_col="dataset",
                value_col="score",
                aggregation="max",
                tools_on="columns",
                output_path=output_dir / "performance_columns.pdf",
            )
            plt.close(figure)

            ranks = trust.rank_tools(
                value_col="score",
                tool_col="tool",
                aggregation="mean",
                category_col="dataset",
            )
            figure, rank_result = trust.plot.rank_correlation(
                rank_df=ranks,
                dataset_col="dataset",
                tool_col="tool",
                output_path=output_dir / "rank_correlation.pdf",
            )
            self.assertEqual(rank_result["correlation"].shape, (3, 3))
            plt.close(figure)

            figure, embedding_result = trust.plot.embedding_umap(
                embedding_col="embedding",
                color_col="label",
                n_neighbors=5,
                output_path=output_dir / "embedding.pdf",
            )
            self.assertEqual(len(embedding_result["embedding"]), len(trust.df))
            plt.close(figure)

            resource_df = pd.DataFrame(
                {
                    "tool": ["Tool A", "Tool B", "Tool C", "Tool D"],
                    "time": [0, 1, 2, np.nan],
                    "memory": [4, 8, 12, 16],
                }
            )
            figure, _ = trust.plot.resource_usage(
                resource_df=resource_df,
                tool_col="tool",
                time_col="time",
                memory_col="memory",
                output_path=output_dir / "resources.pdf",
            )
            plt.close(figure)

            figure, bubble_result = trust.plot.metric_bubbles(
                value_cols=["score"],
                tool_col="tool",
                output_path=output_dir / "metric_bubbles.pdf",
            )
            self.assertEqual(bubble_result["matrix"].shape, (1, 4))
            plt.close(figure)

            figure, bar_result = trust.plot.metric_bars(
                value_col="score",
                tool_col="tool",
                aggregation="mean",
                category_col="dataset",
                output_path=output_dir / "bars.pdf",
            )
            self.assertEqual(len(bar_result["summary"]), 12)
            plt.close(figure)

            blat_summary = pd.DataFrame(
                {
                    "query_group": ["Tool A", "Tool B", "Tool C", "Tool D"],
                    "n_query_matched": [93, 85, 23, 6],
                    "n_query_total": [100, 100, 100, 100],
                }
            )
            figure, blat_result = trust.plot.blat_match(
                summary_df=blat_summary,
                output_path=output_dir / "blat.pdf",
            )
            self.assertAlmostEqual(
                blat_result["proportions"].iloc[0]["matched_percent"],
                93.0,
            )
            plt.close(figure)

            expected = {
                "performance_rows.pdf",
                "performance_columns.pdf",
                "rank_correlation.pdf",
                "embedding.pdf",
                "resources.pdf",
                "bars.pdf",
                "blat.pdf",
            }
            self.assertEqual({path.name for path in output_dir.glob("*.pdf")}, expected)
            for filename in expected:
                self.assert_pdf(output_dir / filename)


if __name__ == "__main__":
    unittest.main()
