"""Tests de las métricas de evaluación."""

import math

from skfold_kge.metrics import (
    compute_filtered_ranks,
    compute_metrics_from_ranks,
    f1_pairwise,
)


def test_metrics_from_known_ranks():
    ranks = [1, 2, 3, 10, 11]
    m = compute_metrics_from_ranks(ranks)
    expected_mrr = (1 + 1 / 2 + 1 / 3 + 1 / 10 + 1 / 11) / 5
    assert math.isclose(m["MRR"], expected_mrr, rel_tol=1e-9)
    assert math.isclose(m["Hits@1"], 1 / 5)
    assert math.isclose(m["Hits@3"], 3 / 5)
    assert math.isclose(m["Hits@10"], 4 / 5)


def test_metrics_empty():
    m = compute_metrics_from_ranks([])
    assert m["MRR"] == 0.0


def test_filtered_ranks_true_triple_ranks_first():
    entities = ["A", "B", "C"]
    true = ("A", "rel", "B")

    def score_fn(s, r, o):
        return 1.0 if (s, r, o) == true else 0.0

    filter_triples = {true}
    ranks_obj = compute_filtered_ranks(
        [true], entities, score_fn, "object", filter_triples
    )
    ranks_subj = compute_filtered_ranks(
        [true], entities, score_fn, "subject", filter_triples
    )
    assert ranks_obj == [1]
    assert ranks_subj == [1]


def test_f1_pairwise_perfect_classifier():
    entities = ["A", "B", "C", "D"]
    positives = [("A", "rel", "B"), ("C", "rel", "D")]
    pos_set = set(positives)

    def score_fn(s, r, o):
        # Puntúa alto solo las positivas -> clasificador perfecto.
        return 1.0 if (s, r, o) in pos_set else 0.0

    res = f1_pairwise(positives, entities, score_fn, pos_set, seed=1)
    assert res["F1"] == 1.0
    assert res["Precision"] == 1.0 and res["Recall"] == 1.0
