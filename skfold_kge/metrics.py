"""Métricas de evaluación para enlace-predicción (KGE), sin dependencias pesadas.

Incluye el *filtered ranking* estándar (MRR, Hits@K) y un **F1 pairwise**
genérico que recibe una función de puntuación ``score_fn(s, r, o) -> float``,
de modo que sirve para cualquier modelo sin acoplarse a torch/pykeen.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

Triple = Tuple[object, object, object]
ScoreFn = Callable[[object, object, object], float]


def compute_filtered_ranks(
    test_triples: Sequence[Triple],
    entities_list: Sequence[object],
    score_fn: ScoreFn,
    corrupt_side: str,
    filter_triples: Set[Triple],
) -> List[int]:
    """Rango filtrado de cada tripleta corrompiendo sujeto u objeto.

    Para cada tripleta de prueba se generan candidatos sustituyendo el lado
    indicado por todas las entidades, se excluyen las tripletas verdaderas
    conocidas (*filtered setting*) y se ordena por puntuación descendente.

    Parameters
    ----------
    test_triples : sequence of (s, r, o)
    entities_list : sequence
        Universo de entidades candidatas.
    score_fn : callable
        ``score_fn(s, r, o) -> float`` (mayor = más plausible).
    corrupt_side : {'subject', 'object'}
    filter_triples : set of (s, r, o)
        Tripletas verdaderas a excluir de los candidatos.

    Returns
    -------
    list of int
        Rango (1 = mejor) de la tripleta verdadera por cada caso de prueba.
    """
    if corrupt_side not in ("subject", "object"):
        raise ValueError("corrupt_side debe ser 'subject' u 'object'.")

    ranks: List[int] = []
    for s, r, o in test_triples:
        true_score = score_fn(s, r, o)
        candidate_scores: List[float] = []
        for entity in entities_list:
            corrupted = (entity, r, o) if corrupt_side == "subject" else (s, r, entity)
            if corrupted not in filter_triples:
                candidate_scores.append(score_fn(*corrupted))
        all_scores = sorted(candidate_scores + [true_score], reverse=True)
        ranks.append(all_scores.index(true_score) + 1)
    return ranks


def compute_metrics_from_ranks(ranks: Sequence[int]) -> Dict[str, float]:
    """Agrega rangos en MRR y Hits@1/3/10."""
    if not ranks:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    mrr = hits1 = hits3 = hits10 = 0.0
    for rank in ranks:
        mrr += 1.0 / rank
        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        if rank <= 10:
            hits10 += 1
    n = len(ranks)
    return {
        "MRR": mrr / n,
        "Hits@1": hits1 / n,
        "Hits@3": hits3 / n,
        "Hits@10": hits10 / n,
    }


def f1_pairwise(
    test_triples: Sequence[Triple],
    entities_list: Sequence[object],
    score_fn: ScoreFn,
    filter_triples: Set[Triple],
    seed: int = 42,
    max_tries: int = 10,
) -> Dict[str, float]:
    """F1 pairwise: clasifica positivo vs. negativo muestreado 1:1.

    Por cada tripleta positiva se genera una negativa corrompiendo el objeto
    (evitando tripletas verdaderas). Se predice "positivo" cuando
    ``score(pos) > score(neg)``. En el caso balanceado 1:1, F1 = Precision =
    Recall = exactitud pairwise.

    Returns
    -------
    dict
        ``{"Precision", "Recall", "F1"}``.
    """
    rng = np.random.RandomState(seed)
    entities = np.asarray(list(entities_list), dtype=object)
    if len(entities) == 0:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    correct = 0
    total = 0
    for s, r, o in test_triples:
        neg_o = o
        for _ in range(max_tries):
            cand = entities[rng.randint(len(entities))]
            if cand != o and (s, r, cand) not in filter_triples:
                neg_o = cand
                break
        if neg_o == o:
            continue
        total += 1
        if score_fn(s, r, o) > score_fn(s, r, neg_o):
            correct += 1

    if total == 0:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    f1 = correct / total
    return {"Precision": round(f1, 4), "Recall": round(f1, 4), "F1": round(f1, 4)}
