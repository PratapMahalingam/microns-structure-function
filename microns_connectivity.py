"""Structure-function analysis of the MICrONS mouse visual cortex connectome.

Question: do synaptically connected neurons respond more similarly to natural
movies than expected from the network's degree structure alone, and than
expected from axon-dendrite proximity alone?

Data: MICrONS functional connectomics release (Ding et al., 2025, Nature,
doi:10.1038/s41586-025-08840-3), node_data_v1.pkl / edge_data_v1.pkl from the
public BossDB bucket. Run ``python download_microns.py`` first.

Usage:
    python microns_connectivity.py --data-dir data --out results
    python microns_connectivity.py --synthetic --out results_synthetic   # smoke test
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------


def zscore_rows(responses: np.ndarray) -> np.ndarray:
    """Z-score each row (neuron) across its response vector.

    Rows with zero variance become all-NaN so they cannot silently produce r=0.
    """
    responses = np.asarray(responses, dtype=np.float64)
    mu = responses.mean(axis=1, keepdims=True)
    sd = responses.std(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (responses - mu) / sd
    z[(sd == 0).ravel()] = np.nan
    return z


def pair_similarity(z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Pearson r between rows i[k] and j[k] of a row-z-scored matrix."""
    return np.einsum("ij,ij->i", z[i], z[j]) / z.shape[1]


# --------------------------------------------------------------------------
# Null models
# --------------------------------------------------------------------------


def degree_preserving_shuffle(pre: np.ndarray, post: np.ndarray,
                              rng: np.random.Generator,
                              max_repair_iter: int = 100_000) -> np.ndarray:
    """Randomise a directed graph while keeping every node's in- and out-degree.

    The presynaptic column is kept and the postsynaptic column is permuted,
    which preserves both degree sequences exactly. Self-loops and duplicate
    edges created by the permutation are then removed by swapping the targets
    of offending edges with randomly chosen edges (a double-edge swap), which
    also preserves degrees. Returns the new postsynaptic array aligned to `pre`.
    """
    pre = np.asarray(pre)
    post = rng.permutation(np.asarray(post))
    m = len(pre)

    def bad_mask(p):
        keys = pre.astype(np.int64) * (1 << 31) + p.astype(np.int64)
        dup = pd.Series(keys).duplicated(keep="first").to_numpy()
        return (pre == p) | dup

    bad = np.flatnonzero(bad_mask(post))
    edges = set(zip(pre.tolist(), post.tolist()))
    it = 0
    while bad.size:
        it += 1
        if it > max_repair_iter:
            raise RuntimeError("could not repair shuffled graph; graph too dense")
        a = bad[rng.integers(bad.size)]
        b = rng.integers(m)
        if a == b:
            continue
        pa, qa, pb, qb = pre[a], post[a], pre[b], post[b]
        # proposed edges after swap: pa->qb, pb->qa
        if pa == qb or pb == qa:
            continue
        if (pa, qb) in edges or (pb, qa) in edges:
            continue
        post[a], post[b] = qb, qa
        edges = set(zip(pre.tolist(), post.tolist()))
        bad = np.flatnonzero(bad_mask(post))
    return post


def proximity_matched_shuffle(pre: np.ndarray, candidates: dict,
                              rng: np.random.Generator) -> np.ndarray:
    """For each presynaptic neuron, redraw its targets from its proximal partners.

    `candidates[p]` is a tuple (post_ids, weights) listing every neuron whose
    dendrite co-travels with p's axon (connected or not). Each neuron keeps its
    out-degree; targets are drawn without replacement with probability
    proportional to `weights` (axon-dendrite co-travel length).
    Returns a new post array aligned to `pre`.
    """
    pre = np.asarray(pre)
    out = np.empty_like(pre)
    for p in np.unique(pre):
        idx = np.flatnonzero(pre == p)
        ids, w = candidates[p]
        prob = w / w.sum()
        out[idx] = rng.choice(ids, size=idx.size, replace=False, p=prob)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@dataclass
class NullResult:
    name: str
    observed_mean: float
    null_means: np.ndarray          # (n_shuffles,)
    null_pooled: np.ndarray         # all shuffled-pair similarities
    per_pre_null_mean: np.ndarray   # (n_pre,) expected r per presyn. neuron

    @property
    def p_perm(self) -> float:
        """One-sided permutation p: P(null mean >= observed mean)."""
        n = self.null_means.size
        return float((1 + np.sum(self.null_means >= self.observed_mean)) / (1 + n))

    @property
    def z(self) -> float:
        return float((self.observed_mean - self.null_means.mean()) / self.null_means.std(ddof=1))


def cliffs_delta_from_u(u: float, n1: int, n2: int) -> float:
    """Cliff's delta (= rank-biserial correlation) from a Mann-Whitney U of sample 1."""
    return 2.0 * u / (n1 * n2) - 1.0


def run_null(name, z, pre_idx, post_idx, observed, shuffle_fn, n_shuffles, rng):
    uniq, inv = np.unique(pre_idx, return_inverse=True)
    null_means = np.empty(n_shuffles)
    pooled = np.empty((n_shuffles, len(pre_idx)), dtype=np.float32)
    per_pre = np.zeros(len(uniq))
    counts = np.bincount(inv)
    for s in range(n_shuffles):
        post_s = shuffle_fn(rng)
        r = pair_similarity(z, pre_idx, post_s)
        null_means[s] = np.nanmean(r)
        pooled[s] = r
        per_pre += np.bincount(inv, weights=np.nan_to_num(r)) / counts
    return NullResult(name, float(np.nanmean(observed)), null_means,
                      pooled.ravel(), per_pre / n_shuffles)


def summarise(res: NullResult, observed: np.ndarray, pre_idx: np.ndarray) -> dict:
    obs = observed[~np.isnan(observed)]
    nul = res.null_pooled[~np.isnan(res.null_pooled)]
    u, p_mw = stats.mannwhitneyu(obs, nul, alternative="greater")
    uniq, inv = np.unique(pre_idx, return_inverse=True)
    obs_per_pre = np.bincount(inv, weights=observed) / np.bincount(inv)
    diff = obs_per_pre - res.per_pre_null_mean
    w = stats.wilcoxon(diff, alternative="greater")
    return {
        "null": res.name,
        "n_pairs": int(obs.size),
        "n_presynaptic_neurons": int(uniq.size),
        "n_shuffles": int(res.null_means.size),
        "observed_mean_r": res.observed_mean,
        "observed_median_r": float(np.median(obs)),
        "null_mean_r": float(res.null_means.mean()),
        "null_mean_r_95ci": [float(np.percentile(res.null_means, 2.5)),
                             float(np.percentile(res.null_means, 97.5))],
        "null_median_r": float(np.median(nul)),
        "difference_in_mean_r": float(res.observed_mean - res.null_means.mean()),
        "permutation_p_one_sided": res.p_perm,
        "permutation_z": res.z,
        "mann_whitney_U": float(u),
        "mann_whitney_p_one_sided": float(p_mw),
        "cliffs_delta": cliffs_delta_from_u(u, obs.size, nul.size),
        "per_neuron_wilcoxon_p_one_sided": float(w.pvalue),
        "per_neuron_fraction_above_null": float(np.mean(diff > 0)),
    }


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    nodes: pd.DataFrame       # one row per neuron, index 0..N-1, column nucleus_id
    responses: np.ndarray     # (N, T)
    edges: pd.DataFrame       # connected pairs: pre_idx, post_idx, n_synapses, synapse_size, ...
    proximal: pd.DataFrame    # all co-travelling pairs (connected + not): pre_idx, post_idx, dend_len
    label: str                # "MICrONS" or "SYNTHETIC"


def load_microns(data_dir: Path) -> Dataset:
    nodes = pd.read_pickle(data_dir / "node_data_v1.pkl")
    edges = pd.read_pickle(data_dir / "edge_data_v1.pkl")
    nodes = nodes.reset_index(drop=True)
    responses = np.stack(nodes["in_vivo_mean_resp"].to_numpy()).astype(np.float64)
    idx = pd.Series(np.arange(len(nodes)), index=nodes["nucleus_id"].to_numpy())
    keep = ["pre_nucleus_id", "post_nucleus_id", "n_synapses", "synapse_size",
            "dend_len", "pre_layer", "post_layer", "pre_hva", "post_hva",
            "in_vivo_sig_corr"]
    sub = edges.loc[edges["population"].isin(["Connected", "ADP"]), keep + ["population"]].copy()
    sub["pre_idx"] = idx.loc[sub["pre_nucleus_id"]].to_numpy()
    sub["post_idx"] = idx.loc[sub["post_nucleus_id"]].to_numpy()
    conn = sub[sub["population"] == "Connected"].reset_index(drop=True)
    slim = nodes.drop(columns=[c for c in nodes.columns if nodes[c].dtype == object])
    return Dataset(slim, responses, conn, sub.reset_index(drop=True), "MICrONS")


def make_synthetic(n_neurons=500, n_pre=40, n_timepoints=120, p_proximal=0.25,
                   p_connect_given_proximal=0.25, coupling=0.3, seed=0) -> Dataset:
    """Synthetic stand-in with the same structure as the MICrONS tables.

    Neurons have Gaussian response vectors that mix a shared latent component
    with private noise. Connected pairs are made more similar by `coupling`.
    This exists only so the pipeline and CI can run without the 500 MB download.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_neurons, 8))
    basis = rng.standard_normal((8, n_timepoints))
    resp = latent @ basis + rng.standard_normal((n_neurons, n_timepoints)) * 2.0
    pre_ids = rng.choice(n_neurons, n_pre, replace=False)
    rows = []
    for p in pre_ids:
        prox = rng.random(n_neurons) < p_proximal
        prox[p] = False
        for q in np.flatnonzero(prox):
            conn = rng.random() < p_connect_given_proximal
            rows.append((p, q, conn, rng.exponential(0.03) + 1e-4))
    df = pd.DataFrame(rows, columns=["pre_idx", "post_idx", "connected", "dend_len"])
    # make connected targets more similar to their presynaptic partner
    for p, grp in df[df.connected].groupby("pre_idx"):
        resp[grp.post_idx.to_numpy()] += coupling * resp[p]
    df["population"] = np.where(df.connected, "Connected", "ADP")
    n_syn = np.where(df.connected, 1 + rng.poisson(0.15, len(df)), 0)
    df["n_synapses"] = n_syn
    df["synapse_size"] = np.where(df.connected, rng.lognormal(8.6, 0.9, len(df)).round(), 0)
    layers = rng.choice(["L2/3", "L4", "L5"], n_neurons, p=[0.5, 0.3, 0.2])
    df["pre_layer"] = layers[df.pre_idx]
    df["post_layer"] = layers[df.post_idx]
    df["pre_hva"] = "V1"
    df["post_hva"] = "V1"
    nodes = pd.DataFrame({"nucleus_id": np.arange(n_neurons), "layer": layers,
                          "brain_area": "V1"})
    conn = df[df.connected].drop(columns="connected").reset_index(drop=True)
    return Dataset(nodes, resp, conn, df.drop(columns="connected"), "SYNTHETIC")


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def analyse(ds: Dataset, edges: pd.DataFrame, n_shuffles: int, seed: int):
    z = zscore_rows(ds.responses)
    pre = edges["pre_idx"].to_numpy()
    post = edges["post_idx"].to_numpy()
    observed = pair_similarity(z, pre, post)

    rng = np.random.default_rng(seed)
    deg = run_null("degree-preserving shuffle", z, pre, post, observed,
                   lambda r: degree_preserving_shuffle(pre, post, r),
                   n_shuffles, rng)

    prox = ds.proximal[ds.proximal["pre_idx"].isin(np.unique(pre))]
    if "post_layer" in edges and edges is not ds.edges:
        # restrict candidate targets to the same post-layer set as `edges`
        prox = prox[prox["post_layer"].isin(edges["post_layer"].unique())
                    & prox["post_hva"].isin(edges["post_hva"].unique())]
    candidates = {p: (g["post_idx"].to_numpy(), g["dend_len"].to_numpy())
                  for p, g in prox.groupby("pre_idx")}
    prox_res = run_null("proximity-matched shuffle", z, pre, post, observed,
                        lambda r: proximity_matched_shuffle(pre, candidates, r),
                        n_shuffles, rng)
    return observed, deg, prox_res


def strength_relation(edges: pd.DataFrame, observed: np.ndarray) -> dict:
    out = {}
    for col in ["synapse_size", "n_synapses"]:
        rho, p = stats.spearmanr(edges[col], observed, nan_policy="omit")
        out[col] = {"spearman_rho": float(rho), "p": float(p)}
    multi = edges["n_synapses"] > 1
    if multi.any() and (~multi).any():
        u, p = stats.mannwhitneyu(observed[multi], observed[~multi], alternative="greater")
        out["multi_vs_single_synapse"] = {
            "n_multi": int(multi.sum()), "n_single": int((~multi).sum()),
            "mean_r_multi": float(observed[multi].mean()),
            "mean_r_single": float(observed[~multi].mean()),
            "mann_whitney_p_one_sided": float(p),
            "cliffs_delta": cliffs_delta_from_u(u, int(multi.sum()), int((~multi).sum())),
        }
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_figures(ds, edges, observed, deg, prox_res, out: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    banner = "" if ds.label == "MICrONS" else "SYNTHETIC SMOKE TEST - not real data\n"
    colors = {"obs": "#1f4e79", "deg": "#9a9a9a", "prox": "#d08a2c"}

    # 1. similarity distributions
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-0.4, 1.0, 71)
    ax.hist(deg.null_pooled, bins, density=True, histtype="stepfilled", alpha=0.35,
            color=colors["deg"], label=f"degree-preserving null ({deg.null_means.size} shuffles)")
    ax.hist(prox_res.null_pooled, bins, density=True, histtype="step", lw=1.8,
            color=colors["prox"], label="proximity-matched null")
    ax.hist(observed, bins, density=True, histtype="step", lw=2.2,
            color=colors["obs"], label=f"connected pairs (n={observed.size:,})")
    ax.set_xlabel("response similarity (Pearson r, natural movies)")
    ax.set_ylabel("density")
    ax.set_title(f"{banner}Connected vs shuffled pairs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"similarity_distributions{tag}.png", dpi=150)
    plt.close(fig)

    # 2. null of the mean
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, res, c in zip(axes, [deg, prox_res], [colors["deg"], colors["prox"]]):
        ax.hist(res.null_means, 40, color=c, alpha=0.8)
        ax.axvline(res.observed_mean, color=colors["obs"], lw=2, label="observed")
        floor = 1 / (1 + res.null_means.size)
        ptxt = f"p <= {floor:.4f} (no shuffle reached observed)" if res.p_perm <= floor else f"p = {res.p_perm:.4f}"
        ax.set_title(f"{res.name}\n{ptxt}, one-sided", fontsize=9)
        ax.set_xlabel("mean r across pairs")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("shuffles")
    if banner:
        fig.suptitle(banner.strip(), color="crimson")
    fig.tight_layout()
    fig.savefig(out / f"null_mean_distributions{tag}.png", dpi=150)
    plt.close(fig)

    # 3. connection strength vs similarity
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    size = edges["synapse_size"].to_numpy().astype(float)
    axes[0].scatter(size, observed, s=4, alpha=0.25, color=colors["obs"], lw=0)
    axes[0].set_xscale("log")
    qs = np.quantile(size, np.linspace(0, 1, 11))
    mids, means = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (size >= lo) & (size <= hi)
        mids.append(np.sqrt(lo * hi)); means.append(observed[m].mean())
    axes[0].plot(mids, means, "o-", color="black", ms=4, label="decile mean")
    rho, _ = stats.spearmanr(size, observed)
    axes[0].set_title(f"Spearman rho = {rho:.3f}", fontsize=9)
    axes[0].set_xlabel("total synapse size (cleft volume, a.u.)")
    axes[0].set_ylabel("response similarity (r)")
    axes[0].legend(frameon=False, fontsize=8)
    ns = edges["n_synapses"].clip(upper=4).to_numpy()
    groups = [observed[ns == k] for k in range(1, 5) if (ns == k).any()]
    labels = [("4+" if k == 4 else str(k)) + f"\n(n={(ns == k).sum()})" for k in range(1, 5) if (ns == k).any()]
    axes[1].boxplot(groups, tick_labels=labels, showfliers=False)
    axes[1].set_xlabel("synapses per connection")
    axes[1].set_title("similarity by synapse count", fontsize=9)
    if banner:
        fig.suptitle(banner.strip(), color="crimson")
    fig.tight_layout()
    fig.savefig(out / f"strength_vs_similarity{tag}.png", dpi=150)
    plt.close(fig)

    # 4. connectivity graph among presynaptic (proofread) neurons
    pres = set(edges["pre_idx"])
    sub = edges[edges["post_idx"].isin(pres)]
    sub_r = observed[edges["post_idx"].isin(pres).to_numpy()]
    if len(sub) < 20:   # fall back to the ego network of the 3 highest-degree neurons
        top = edges["pre_idx"].value_counts().index[:3]
        m = edges["pre_idx"].isin(top).to_numpy()
        sub, sub_r = edges[m], observed[m]
    g = nx.DiGraph()
    for (a, b), r in zip(sub[["pre_idx", "post_idx"]].to_numpy(), sub_r):
        g.add_edge(int(a), int(b), r=float(r))
    pos = nx.spring_layout(g, seed=1, k=0.6 / np.sqrt(max(g.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    er = np.array([d["r"] for *_, d in g.edges(data=True)])
    lim = max(0.3, np.nanpercentile(np.abs(er), 98))
    layer_col = {"L2/3": "#2a9d8f", "L4": "#e9c46a", "L5": "#8d5a97"}
    layers = ds.nodes["layer"].astype(str).to_numpy()
    nodelist = list(g.nodes())
    nx.draw_networkx_nodes(g, pos, nodelist=nodelist, node_size=22, ax=ax,
                           node_color=[layer_col.get(layers[v], "#444444") for v in nodelist])
    for lab, c in layer_col.items():
        ax.scatter([], [], color=c, s=22, label=lab)
    ax.scatter([], [], color="#444444", s=22, label="no layer label")
    ax.legend(title="layer", frameon=False, fontsize=8, loc="lower left")
    ec = nx.draw_networkx_edges(g, pos, edge_color=er, edge_cmap=plt.cm.RdBu_r,
                                edge_vmin=-lim, edge_vmax=lim, width=1.2, arrows=True,
                                arrowsize=6, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax, shrink=0.7, label="response similarity (r)")
    ax.set_title(f"{banner}Connections among proofread presynaptic neurons "
                 f"({g.number_of_nodes()} neurons, {g.number_of_edges()} edges)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"connectivity_graph{tag}.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic data (smoke test, NOT real results)")
    ap.add_argument("--n-shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    ds = make_synthetic() if args.synthetic else load_microns(args.data_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    subsets = {"all": ds.edges}
    l23 = ds.edges[(ds.edges["pre_layer"] == "L2/3") & (ds.edges["post_layer"] == "L2/3")]
    if len(l23) >= 50:
        subsets["L23_to_L23"] = l23.reset_index(drop=True)

    report = {"data": ds.label,
              "warning": None if ds.label == "MICrONS" else
              "SYNTHETIC SMOKE TEST - replace with real MICrONS data; these numbers mean nothing",
              "n_neurons_with_responses": int(len(ds.nodes)),
              "response_vector_length": int(ds.responses.shape[1]),
              "subsets": {}}
    for name, edges in subsets.items():
        print(f"[{ds.label}] subset={name}: {len(edges)} connected pairs, "
              f"{edges['pre_idx'].nunique()} presynaptic neurons, {args.n_shuffles} shuffles each")
        observed, deg, prox_res = analyse(ds, edges, args.n_shuffles, args.seed)
        entry = {"degree_preserving": summarise(deg, observed, edges["pre_idx"].to_numpy()),
                 "proximity_matched": summarise(prox_res, observed, edges["pre_idx"].to_numpy()),
                 "connection_strength": strength_relation(edges, observed)}
        if "in_vivo_sig_corr" in edges:
            entry["check_max_abs_diff_vs_released_sig_corr"] = float(
                np.nanmax(np.abs(observed - edges["in_vivo_sig_corr"].to_numpy())))
        report["subsets"][name] = entry
        if not args.no_figures:
            make_figures(ds, edges, observed, deg, prox_res, args.out,
                         "" if name == "all" else f"_{name}")
        for k in ("degree_preserving", "proximity_matched"):
            s = entry[k]
            print(f"  {k:18s} obs mean r={s['observed_mean_r']:.4f}  null={s['null_mean_r']:.4f}  "
                  f"perm p={s['permutation_p_one_sided']:.4g}  Cliff's d={s['cliffs_delta']:.3f}  "
                  f"per-neuron Wilcoxon p={s['per_neuron_wilcoxon_p_one_sided']:.3g}")

    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 'results.json'}")
    return report


if __name__ == "__main__":
    main()
