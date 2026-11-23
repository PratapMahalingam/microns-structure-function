import json

from microns_connectivity import main


def test_synthetic_end_to_end(tmp_path):
    report = main(["--synthetic", "--n-shuffles", "20", "--out", str(tmp_path)])
    assert report["data"] == "SYNTHETIC"
    assert "SYNTHETIC" in report["warning"]
    s = report["subsets"]["all"]["degree_preserving"]
    # synthetic data has a planted effect, so the pipeline must detect it
    assert s["observed_mean_r"] > s["null_mean_r"]
    assert s["permutation_p_one_sided"] <= 1 / 21 + 1e-12
    assert json.loads((tmp_path / "results.json").read_text())["data"] == "SYNTHETIC"
    for f in ["similarity_distributions.png", "null_mean_distributions.png",
              "strength_vs_similarity.png", "connectivity_graph.png"]:
        assert (tmp_path / f).exists()
