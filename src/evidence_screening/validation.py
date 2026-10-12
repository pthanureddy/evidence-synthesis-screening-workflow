"""Fixed holdout evaluation against released final-inclusion reviewer labels."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import ImportResult, write_import


def wilson(successes: int, total: int) -> list[float] | None:
    if not 0 <= successes <= total:
        raise ValueError("invalid count interval")
    if total == 0:
        return None
    z, proportion = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def binary_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("aligned nonempty labels/scores required")
    if (
        not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("binary labels and finite bounded scores required")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    labels = labels.astype(int)
    predictions = scores >= threshold
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "n": labels.size,
        "included": int(labels.sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_wilson_95": wilson(tp, tp + fp),
        "recall_wilson_95": wilson(tp, tp + fn),
        "high_priority_fraction": float(predictions.mean()),
        "average_precision": float(average_precision_score(labels, scores))
        if labels.sum()
        else None,
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def choose_threshold(labels, scores, target_recall=0.9):
    if not 0 < target_recall <= 1:
        raise ValueError("target recall must be in (0, 1]")
    binary_metrics(labels, scores, 0.5)
    positive_scores = np.asarray(scores, dtype=float)[np.asarray(labels, dtype=int) == 1]
    if not len(positive_scores):
        raise ValueError("validation set requires positive reviewer labels")
    required = math.ceil(target_recall * len(positive_scores))
    return float(np.sort(positive_scores)[::-1][required - 1])


def benchmark(result: ImportResult, output: Path, seed=42, target_recall=0.9):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    frame = result.frame.loc[result.frame.label_included.notna()].copy().reset_index(drop=True)
    labels = frame.label_included.astype(int)
    if len(labels) < 40 or labels.value_counts().min() < 10 or labels.nunique() != 2:
        raise ValueError("benchmark needs at least 40 records and 10 per class after quarantine")
    indices = np.arange(len(frame))
    training, remaining = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, random_state=seed, stratify=labels.iloc[remaining]
    )
    assert not (
        set(training) & set(validation) or set(training) & set(test) or set(validation) & set(test)
    )
    text = frame.title + "\n" + frame.abstract
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=15000, sublinear_tf=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed
                ),
            ),
        ]
    )
    model.fit(text.iloc[training], labels.iloc[training])
    validation_scores = model.predict_proba(text.iloc[validation])[:, 1]
    threshold = choose_threshold(labels.iloc[validation], validation_scores, target_recall)
    test_scores = model.predict_proba(text.iloc[test])[:, 1]
    test_labels = labels.iloc[test].to_numpy()
    metrics = binary_metrics(test_labels, test_scores, threshold)
    slices = {}
    present = frame.iloc[test].abstract.str.len().to_numpy() > 0
    for name, mask in (("abstract_present", present), ("abstract_missing", ~present)):
        slices[name] = (
            binary_metrics(test_labels[mask], test_scores[mask], threshold)
            if mask.any()
            else {"n": 0}
        )
    write_import(result, output)
    split = {
        name: frame.iloc[ids].group_id.tolist()
        for name, ids in (("train", training), ("validation", validation), ("test", test))
    }
    splits = {identifier: name for name, ids in split.items() for identifier in ids}
    queue = result.frame[
        ["group_id", "record_id", "title", "abstract", "year", "doi", "source_ids"]
    ].copy()
    all_scores = model.predict_proba(result.frame.title + "\n" + result.frame.abstract)[:, 1]
    queue["screening_score"] = all_scores
    queue["priority"] = np.where(all_scores >= threshold, "high", "standard")
    queue["status"] = "needs_manual_review"
    queue["benchmark_split"] = queue.group_id.map(splits).fillna("unlabelled_or_conflict")
    queue = queue.sort_values(["screening_score", "group_id"], ascending=[False, True])
    queue.to_csv(output / "review_queue.csv", index=False)
    template = queue[["group_id", "record_id", "title", "abstract"]].copy()
    for field in ("reviewer", "stage", "decision", "reason"):
        template[field] = ""
    template["source_sha256"] = result.report["source_sha256"]
    template.to_csv(output / "manual_review_template.csv", index=False)
    errors = frame.iloc[test][["group_id", "record_id", "title"]].copy()
    errors["original_reviewer_label"] = test_labels
    errors["screening_score"] = test_scores
    errors["predicted_priority_positive"] = test_scores >= threshold
    errors["error"] = np.where(
        (test_scores >= threshold) == test_labels,
        "none",
        np.where(test_labels == 1, "false_negative", "false_positive"),
    )
    errors.to_csv(output / "holdout_predictions.csv", index=False)
    report = {
        "source_sha256": result.report["source_sha256"],
        "seed": seed,
        "threshold": threshold,
        "target_validation_recall": target_recall,
        "split_ids": split,
        "split_counts": {name: len(ids) for name, ids in split.items()},
        "positive_counts": {
            name: int(frame.loc[frame.group_id.isin(ids), "label_included"].sum())
            for name, ids in split.items()
        },
        "validation": binary_metrics(labels.iloc[validation], validation_scores, threshold),
        "holdout": metrics,
        "holdout_abstract_slices": slices,
        "always_high_priority_baseline": binary_metrics(test_labels, np.ones(len(test)), 0.5),
        "always_standard_priority_baseline": binary_metrics(test_labels, np.zeros(len(test)), 0.5),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "scope": "Retrospective priority benchmark against original reviewers' final-inclusion labels; not initial abstract decisions, environmental validation, calibrated confidence or automated exclusion.",
    }
    # Explicit conversion avoids NumPy scalar leakage into JSON.
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2, default=lambda x: x.item(), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {"model": model, "threshold": threshold, "source_sha256": result.report["source_sha256"]},
        output / "model.joblib",
    )
    return report
