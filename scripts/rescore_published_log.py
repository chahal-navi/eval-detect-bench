"""Re-score the authors' published judge responses with this eval's scorer and metrics.

    uv run python scripts/rescore_published_log.py claude-haiku-4-5

Downloads `measure_logs/<judge>.eval` from the dataset (60-240 MB), runs every
logged judge response through this eval's label parser, and reports how often
it agrees with the authors' scorer and the AUROC it gives. No model is called.

The authors' logs store score 1.0 = real and 0.0 = eval, the opposite of this
eval's votes, so agreement means `vote == 1 - their value`. Some samples in the
published logs errored and were scored by a later retry; their response is in
the score metadata rather than the sample output.
"""

import argparse
import math
from collections import Counter, defaultdict

import numpy as np
from inspect_ai.log import read_eval_log
from inspect_evals_utils.huggingface import hf_hub_download

from eval_detect_bench.dataset import HF_REPO_ID, HF_REVISION
from eval_detect_bench.metrics import PARSE_FAILURE, roc_auc
from eval_detect_bench.probes import PAPER_AUROC, load_probes
from eval_detect_bench.scorer import LABEL_RE, extract_last, label_to_vote

# Log file names in the dataset's measure_logs/ folder.
PUBLISHED_LOGS = {
    "claude-haiku-4-5": "claude-haiku-4-5.eval",
    "claude-sonnet-4-6": "claude-sonnet-4-6.eval",
    "claude-opus-4-6": "claude-opus-4-6.eval",
    "gpt-4.1": "gpt-4.1.eval",
    "gpt-5.1": "gpt-5.1.eval",
    "gpt-5.4": "gpt-5.4.eval",
    "gemini-2.5-pro": "gemini-2.5-pro.eval",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview.eval",
    "deepseek-v3.2": "deepseek-v3.2.eval",
    "glm-5.1": "glm-5.1.eval",
    "kimi-k2.5": "kimi-k2.5.eval",
    "qwen3.5-397b-a17b": "qwen3.5-397b-a17b.eval",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score a published judge log.")
    parser.add_argument("judge", choices=sorted(PUBLISHED_LOGS))
    args = parser.parse_args()

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"measure_logs/{PUBLISHED_LOGS[args.judge]}",
        revision=HF_REVISION,
    )
    log = read_eval_log(path)
    probes = load_probes()
    agreement: Counter[tuple[float, float]] = Counter()
    votes: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for sample in log.samples or []:
        if not sample.scores:
            continue
        theirs = next(iter(sample.scores.values()))
        response = sample.output.completion or (theirs.metadata or {}).get("response", "")
        label_values = probes[sample.metadata["pq_id"]].label_values
        vote = label_to_vote(extract_last(LABEL_RE, response), response, label_values)
        agreement[float(theirs.as_float()), vote] += 1
        tid = str(sample.metadata["transcript_id"])
        labels[tid] = int(bool(sample.metadata["eval_category"]))
        if vote != PARSE_FAILURE:
            votes[tid].append(vote)

    total = sum(agreement.values())
    agree = sum(
        n
        for (theirs, ours), n in agreement.items()
        if math.isclose(ours, 1.0 - theirs) or theirs == ours == PARSE_FAILURE
    )
    print(f"Published log: {log.eval.model}, {total} scored responses")
    print(f"Our parser agrees with theirs on {agree} / {total}")
    print("(their value, our vote) counts:", dict(agreement))
    tids = sorted(votes)
    auc = roc_auc([labels[t] for t in tids], [float(np.mean(votes[t])) for t in tids])
    print(f"AUROC from our parser: {auc:.4f} over {len(tids)} transcripts")
    print(f"Paper's Figure 6 AUROC: {PAPER_AUROC[args.judge]:.3f}")


if __name__ == "__main__":
    main()
