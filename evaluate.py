"""Replay the cached answers through each routing policy. Makes no API calls.

Usage: python evaluate.py [data_dir]
Prints the tables and writes results.png next to the data.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backends import BACKENDS
from router import choose, embed, routellm_quality

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
FOLDS = 5
ALPHAS = np.arange(0, 20.001, 0.005)  # cost weights swept to trace each curve
BETAS = [0, 0.25, 0.5, 1]             # latency weights shown in the last table


def load():
    qs = json.loads((DATA / "questions.json").read_text())
    resp = {(r["backend"], r["qid"]): r for r in map(json.loads, (DATA / "responses.jsonl").open())}
    backends = [b for b in BACKENDS if any(name == b["name"] for name, _ in resp)]
    # Only questions every backend answered, so all policies are compared on the same set.
    qs = sorted((q for q in qs if all((b["name"], q["qid"]) in resp for b in backends)), key=lambda q: q["qid"])
    if len(qs) < 50 or len(backends) < 2:
        sys.exit(f"Need 2+ backends with 50+ shared answers. Have {len(qs)} from {[b['name'] for b in backends]}")
    grid = lambda f: np.array([[f(b, resp[b["name"], q["qid"]]) for b in backends] for q in qs], dtype=float)
    return (qs, [b["name"] for b in backends],
            grid(lambda b, r: r["correct"]),
            grid(lambda b, r: 1000 * (r["in_tok"] * b["usd_in"] + r["out_tok"] * b["usd_out"]) / 1e6),
            grid(lambda b, r: r["latency_s"]))


def features(qs):
    cache = DATA / "features.npz"
    if cache.exists():
        f = np.load(cache)
        if list(f["qids"]) == [q["qid"] for q in qs]:
            return f["scores"], f["vecs"]
    print(f"encoding {len(qs)} prompts with the RouteLLM BERT router...")
    scores, vecs = embed([q["prompt"] for q in qs])
    np.savez(cache, qids=[q["qid"] for q in qs], scores=scores, vecs=vecs)
    return scores, vecs


def gain_over_random(picks, correct, usd, idx):
    """Mean accuracy gained over random routing at equal cost, on the questions in idx.

    picks holds one row of backend choices per policy setting. Random routing mixes the
    cheapest backend with the most accurate one, so it sits on the line between them.
    """
    cost, acc = usd[idx, picks[:, idx]].mean(1), correct[idx, picks[:, idx]].mean(1)
    solo_cost, solo_acc = usd[idx].mean(0), correct[idx].mean(0)
    lo, hi = solo_cost.argmin(), solo_acc.argmax()
    if lo == hi:  # the cheapest backend is also the most accurate: nothing to route
        return 0.0
    levels = np.linspace(solo_cost[lo], solo_cost[hi], 21)[1:-1]
    random = solo_acc[lo] + (solo_acc[hi] - solo_acc[lo]) * (levels - solo_cost[lo]) / (solo_cost[hi] - solo_cost[lo])
    order = np.argsort(cost)
    return (np.interp(levels, cost[order], acc[order]) - random).mean()


def main():
    qs, names, correct, usd, sec = load()
    scores, _ = features(qs)
    n, k = correct.shape
    rows = np.arange(n)

    # Everything a router knows about a question comes from the other folds.
    # qids are hashes, so striding through the sorted list gives random folds.
    cost, latency = np.zeros((n, k)), np.zeros((n, k))
    quality = {name: np.zeros((n, k)) for name in ("base rates only", "RouteLLM score as published", "RouteLLM score, calibrated")}
    for fold in range(FOLDS):
        test = rows % FOLDS == fold
        train = ~test
        acc = correct[train].mean(0)
        cost[test] = usd[train].mean(0) / usd[train].mean(0).max()
        latency[test] = np.median(sec[train], 0) / np.median(sec[train], 0).max()
        quality["base rates only"][test] = acc
        quality["RouteLLM score as published"][test] = routellm_quality(scores[test], (acc - acc.min()) / (acc.max() - acc.min()))
        for j in range(k):
            # One logistic curve per backend: how its accuracy moves with the router's score.
            model = LogisticRegression().fit(scores[train, None], correct[train, j])
            quality["RouteLLM score, calibrated"][test, j] = model.predict_proba(scores[test, None])[:, 1]

    def frontier(q, beta):
        return np.unique(np.array([choose(q, cost, latency, a, beta) for a in ALPHAS]), axis=0)

    def measure(picks):
        return dict(acc=correct[rows, picks].mean(), usd=usd[rows, picks].mean(), sec=sec[rows, picks].mean(),
                    mix=" / ".join(f"{(picks == j).mean():.0%}" for j in range(k)))

    def line(label, m):
        print(f"{label:<36}{m['acc']:>7.1%}{m['usd']:>10.4f}{m['sec']:>8.2f}   {m['mix']}")

    head = f"{'':<36}{'acc':>7}{'$/1k q':>10}{'sec/q':>8}   mix"
    print(f"\n{n} questions, {FOLDS}-fold cross-validation\nmix = share of questions sent to: " + " / ".join(names))
    print(f"\nSingle backend\n{head}")
    solo = [measure(np.full(n, j)) for j in range(k)]
    for name, m in zip(names, solo):
        line(name, m)
    # Upper bound: the cheapest backend that answers each question correctly.
    oracle = measure(np.argmin(np.where(correct > 0, usd, usd + 1e9), 1))
    line("oracle (cheapest correct backend)", oracle)

    # A useful score is high where weaker backends fail, so these AUCs should be well below 0.5.
    length = np.log([len(q["prompt"]) for q in qs])
    print(f"\nRouteLLM score: correlation with log prompt length {np.corrcoef(scores, length)[0, 1]:+.2f}")
    print("AUC of the score for predicting a correct answer: "
          + ", ".join(f"{name.split('@')[0]} {roc_auc_score(correct[:, j], scores):.2f}" for j, name in enumerate(names)))

    print("\nAccuracy gained over random routing at equal cost (points, 90% bootstrap interval)")
    rng = np.random.default_rng(0)
    fronts = {name: frontier(q, 0) for name, q in quality.items()}
    for name, picks in fronts.items():
        boot = [gain_over_random(picks, correct, usd, rng.integers(0, n, n)) for _ in range(1000)]
        print(f"{name:<36}{100 * gain_over_random(picks, correct, usd, rows):>+6.1f}"
              f"   [{100 * np.percentile(boot, 5):+.1f}, {100 * np.percentile(boot, 95):+.1f}]")

    print(f"\nLatency weight, calibrated router with no cost weight\n{head}")
    for beta in BETAS:
        line(f"beta={beta}", measure(choose(quality["RouteLLM score, calibrated"], cost, latency, 0, beta)))

    plot(names, solo, oracle, {name: [measure(p) for p in picks] for name, picks in fronts.items()}, n)


def plot(names, solo, oracle, fronts, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    lo, hi = min(solo, key=lambda m: m["usd"]), max(solo, key=lambda m: m["acc"])
    ax.plot([lo["usd"], hi["usd"]], [100 * lo["acc"], 100 * hi["acc"]], color=muted, lw=2, ls=(0, (4, 3)), label="Random routing")
    for name, color in (("RouteLLM score as published", "#2a78d6"), ("RouteLLM score, calibrated", "#eb6834")):
        pts = sorted(fronts[name], key=lambda m: m["usd"])
        ax.plot([m["usd"] for m in pts], [100 * m["acc"] for m in pts], color=color, lw=2, label=name)
    for name, m in zip(names, solo):
        ax.plot(m["usd"], 100 * m["acc"], "o", ms=8, color=ink, mec=surface, mew=2, zorder=3)
        dx, dy, ha, va = (0, 9, "right", "bottom") if m is hi else (8, -16 if m is lo else -9, "left", "top")
        ax.annotate(name.replace("@", " on "), (m["usd"], 100 * m["acc"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=8.5, color=ink)
    ax.plot(oracle["usd"], 100 * oracle["acc"], "o", ms=8, mfc=surface, mec=ink, mew=2, zorder=3)
    ax.annotate("Oracle: cheapest backend\nthat answers correctly", (oracle["usd"], 100 * oracle["acc"]),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=8.5, color=ink)
    ax.set_xlabel("Cost per 1,000 questions (USD, list prices)", color=muted, fontsize=9)
    ax.set_ylabel("Accuracy (%)", color=muted, fontsize=9)
    ax.set_title(f"Accuracy against cost on {n} MMLU-Pro questions", loc="left", color=ink, fontsize=11, pad=12)
    ax.set_ylim(bottom=60)
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, labelsize=8.5, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor="#52514e")
    fig.tight_layout()
    fig.savefig(DATA / "results.png", dpi=160, facecolor=surface)


if __name__ == "__main__":
    main()
