import json

import joblib
import numpy as np
import pytest

from evidence_screening.data import import_records
from evidence_screening.validation import benchmark, binary_metrics, choose_threshold, wilson


def test_metrics_and_denominators():
    result = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["precision_wilson_95"][0] < 0.5 < result["precision_wilson_95"][1]


def test_zero_denominators_are_not_perfect_scores():
    result = binary_metrics([0, 0], [0.1, 0.1], 0.5)
    assert result["precision"] is None and result["recall"] is None
    assert result["average_precision"] is None and result["f1"] == 0
    assert wilson(0, 0) is None


@pytest.mark.parametrize(
    "labels,scores,threshold",
    [
        ([], [], 0.5),
        ([1], [np.nan], 0.5),
        ([2], [0.5], 0.5),
        ([0.5], [0.5], 0.5),
        ([1], [2], 0.5),
        ([1], [0.5], np.nan),
        ([1], [0.5], 2),
    ],
)
def test_invalid_metric_inputs(labels, scores, threshold):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores, threshold)


@pytest.mark.parametrize("successes,total", [(-1, 5), (6, 5)])
def test_invalid_wilson_counts(successes, total):
    with pytest.raises(ValueError):
        wilson(successes, total)


def test_wilson_boundaries():
    assert wilson(0, 10)[0] == 0
    assert wilson(10, 10)[1] == pytest.approx(1)


def test_validation_threshold_recall_constraint():
    threshold = choose_threshold([1, 1, 0, 0], [0.9, 0.3, 0.6, 0.1], 1)
    assert threshold == 0.3
    assert binary_metrics([1, 1, 0, 0], [0.9, 0.3, 0.6, 0.1], threshold)["recall"] == 1
    with pytest.raises(ValueError):
        choose_threshold([0, 0], [0.9, 0.5])


@pytest.mark.parametrize("value", [-0.1, 0, 1.1])
def test_invalid_recall_target(value):
    with pytest.raises(ValueError):
        choose_threshold([1, 0], [0.9, 0.2], value)


def test_full_workflow_reproducible_splits_queue_and_no_text_leak(input_csv, tmp_path):
    result = import_records(input_csv)
    first = benchmark(result, tmp_path / "a")
    second = benchmark(result, tmp_path / "b")
    assert first["split_ids"] == second["split_ids"]
    assert first["holdout"] == second["holdout"]
    sets = [set(ids) for ids in first["split_ids"].values()]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    held_out = set(first["split_ids"]["validation"] + first["split_ids"]["test"])
    result.frame.loc[result.frame.group_id.isin(held_out), "abstract"] += " leakagetoken"
    benchmark(result, tmp_path / "c")
    model = joblib.load(tmp_path / "c/model.joblib")["model"]
    assert "leakagetoken" not in model.named_steps["tfidf"].vocabulary_
    import pandas as pd

    queue = pd.read_csv(tmp_path / "c/review_queue.csv")
    assert set(queue.status) == {"needs_manual_review"}
    template = pd.read_csv(tmp_path / "c/manual_review_template.csv")
    assert "label_included" not in template.columns and template.decision.isna().all()
    json.loads((tmp_path / "c/benchmark.json").read_text())


def test_insufficient_classes_rejected(input_csv, tmp_path):
    result = import_records(input_csv)
    result.frame["label_included"] = 0
    with pytest.raises(ValueError):
        benchmark(result, tmp_path / "bad")


@pytest.mark.parametrize("seed", [-1, True, 2**32])
def test_invalid_seed(input_csv, tmp_path, seed):
    with pytest.raises(ValueError):
        benchmark(import_records(input_csv), tmp_path, seed=seed)
