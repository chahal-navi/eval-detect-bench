"""Compare a run of this eval with the paper's published scores for the same judge.

    uv run python scripts/compare_with_paper.py logs/<run>.eval

Prints the run's metrics, then compares it with the authors' per-transcript
P(eval) (`measure_logs/scores.csv` in the dataset) on the transcripts both
scored: AUROC for each, a paired bootstrap interval for the difference, and
the per-transcript correlation.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from inspect_ai.log import read_eval_log
from inspect_evals_utils.huggingface import hf_hub_download

from eval_detect_bench.dataset import HF_REPO_ID, HF_REVISION
from eval_detect_bench.metrics import PARSE_FAILURE, roc_auc
from eval_detect_bench.probes import PAPER_AUROC

SCORER_NAME = "probe_label_scorer"


def run_p_eval(log_path: str) -> tuple[str, dict[str, float], dict[str, int]]:
    """Per-transcript P(eval) and labels from a run of this eval."""
    log = read_eval_log(log_path)
    pool = (log.eval.metadata or {})["transcript_pool"]
    votes: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for sample in log.samples or []:
        score = (sample.scores or {}).get(SCORER_NAME)
        tid = str(sample.metadata["transcript_id"])
        labels[tid] = int(bool(sample.metadata["eval_category"]))
        if score is not None and score.as_float() != PARSE_FAILURE:
            votes[tid].append(score.as_float())
    print(f"Run: {log.eval.model} on pool {pool} ({log.status})")
    if log.results:
        for name, m in log.results.scores[0].metrics.items():
            print(f"  {name:24s} {m.value:.4f}")
    return pool, {t: float(np.mean(v)) for t, v in votes.items()}, labels


def paper_p_eval(pool: str) -> tuple[dict[str, float], dict[str, int]]:
    """The paper's per-transcript P(eval) and labels for the judge of `pool`."""
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename="measure_logs/scores.csv",
        revision=HF_REVISION,
    )
    votes: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["judge"].rsplit("/", 1)[-1] != pool:
                continue
            labels[row["transcript_id"]] = int(row["eval_category"].lower() == "true")
            if float(row["p_eval"]) >= 0:
                votes[row["transcript_id"]].append(float(row["p_eval"]))
    return {t: float(np.mean(v)) for t, v in votes.items()}, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a run with the paper's scores.")
    parser.add_argument("log", help="An .eval log from this eval (full run, paper probe set).")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pool, ours, labels = run_p_eval(args.log)
    paper, paper_labels = paper_p_eval(pool)
    common = sorted(set(ours) & set(paper))
    only_ours = len(set(ours) - set(paper))
    only_paper = len(set(paper) - set(ours))
    y = np.array([labels[t] for t in common])
    assert all(labels[t] == paper_labels[t] for t in common), "labels disagree"
    a = np.array([ours[t] for t in common])
    b = np.array([paper[t] for t in common])

    print(f"\nTranscripts: run {len(ours)}, paper {len(paper)}, both {len(common)}")
    print(f"  only in run: {only_ours}, only in paper: {only_paper}")
    print(
        f"Paper's Figure 6 AUROC (all its transcripts): {PAPER_AUROC.get(pool, float('nan')):.3f}"
    )
    print(
        f"AUROC on the {len(common)} common transcripts: run {roc_auc(y, a):.4f}, paper {roc_auc(y, b):.4f}"
    )

    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.bootstrap):
        i = rng.integers(0, len(common), len(common))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc(y[i], a[i]) - roc_auc(y[i], b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"Difference (run - paper), paired bootstrap: {np.mean(diffs):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]"
    )
    print(f"Per-transcript P(eval) correlation: r = {np.corrcoef(a, b)[0, 1]:.3f}")
    print(
        f"Mean P(eval) eval / deploy: run {a[y == 1].mean():.3f} / {a[y == 0].mean():.3f}, "
        f"paper {b[y == 1].mean():.3f} / {b[y == 0].mean():.3f}"
    )


if __name__ == "__main__":
    main()
