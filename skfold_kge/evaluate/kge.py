"""Validación cruzada de modelos KGE (TransE, ComplEx, RotatE) con pykeen.

Extra ``[kge]``. Encapsula el bucle del notebook: por cada modelo y fold
entrena en los ``k-1`` folds restantes, evalúa en el fold de prueba con
*filtered ranking* (MRR, Hits@K) y añade un **F1 pairwise**. Devuelve un dict
con promedio y desviación estándar entre folds, listo para
:meth:`IntegrityReport.to_html`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..partition import FoldSet

_METRIC_KEYS = ["MRR", "Hits@1", "Hits@3", "Hits@10", "F1"]


def _require_pykeen():
    try:
        import torch  # noqa: F401
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cross_validate_kge requiere los extras de KGE. Instale con:\n"
            "    pip install skfold-kge[kge]\n"
            f"(causa: {exc})"
        ) from exc
    return torch, pipeline, TriplesFactory


def _triples_array(foldset: "FoldSet", positions: Sequence[int]) -> np.ndarray:
    cols = foldset.triple_columns
    if cols is None:
        raise ValueError(
            "El FoldSet no está en modo tripletas. Indique triple_columns "
            "en el StratifiedPartitioner."
        )
    sub = foldset.clean_df.iloc[list(positions)]
    return sub[list(cols)].astype(str).to_numpy()


def _f1_pairwise_kge(model, test_arr, all_triples_set, tf_global, torch, seed=42):
    """F1 pairwise usando la puntuación vectorizada de pykeen (model.score_hrt)."""
    rng = np.random.RandomState(seed)
    entities = np.array(list(tf_global.entity_to_id.keys()))
    pos_hrts: List[List[int]] = []
    neg_hrts: List[List[int]] = []

    e2i = tf_global.entity_to_id
    r2i = tf_global.relation_to_id
    for s, r, o in test_arr:
        if s not in e2i or r not in r2i or o not in e2i:
            continue
        neg_e = o
        for _ in range(10):
            cand = rng.choice(entities)
            if cand != o and (s, r, cand) not in all_triples_set:
                neg_e = cand
                break
        if neg_e == o:
            continue
        pos_hrts.append([e2i[s], r2i[r], e2i[o]])
        neg_hrts.append([e2i[s], r2i[r], e2i[neg_e]])

    if not pos_hrts:
        return 0.0

    model.eval()
    with torch.no_grad():
        p = model.score_hrt(torch.tensor(pos_hrts, dtype=torch.long)).cpu().numpy().flatten()
        n = model.score_hrt(torch.tensor(neg_hrts, dtype=torch.long)).cpu().numpy().flatten()
    return float(np.mean(p > n))


def cross_validate_kge(
    foldset: "FoldSet",
    models: Sequence[str] = ("TransE", "ComplEx", "RotatE"),
    num_epochs: int = 200,
    embedding_dim: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    compute_f1: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Compara modelos KGE bajo la partición estratificada de ``foldset``.

    Parameters
    ----------
    foldset : FoldSet
        Debe estar en modo tripletas (``triple_columns`` definido).
    models : sequence of str
        Nombres de modelos pykeen.
    num_epochs, embedding_dim, batch_size, lr, seed : hiperparámetros.
    compute_f1 : bool
        Si añade el F1 pairwise por fold.

    Returns
    -------
    dict
        ``{"models", "metric_keys", "avg", "std", "per_fold", "k",
        "epochs", "dim"}``.
    """
    torch, pipeline, TriplesFactory = _require_pykeen()

    all_arr = _triples_array(foldset, range(len(foldset.clean_df)))
    tf_global = TriplesFactory.from_labeled_triples(all_arr)
    all_triples_set = set(map(tuple, all_arr))

    metric_keys = _METRIC_KEYS if compute_f1 else _METRIC_KEYS[:-1]
    avg: Dict[str, Dict[str, float]] = {}
    std: Dict[str, Dict[str, float]] = {}
    per_fold: Dict[str, List[Dict[str, float]]] = {}

    for model_name in models:
        if verbose:
            print(f"\n=== {model_name} (dim={embedding_dim}, {num_epochs} epochs) ===")
        fold_metrics: List[Dict[str, float]] = []
        for i in range(foldset.k):
            test_arr = _triples_array(foldset, foldset.folds[i])
            train_pos = [p for j in range(foldset.k) if j != i for p in foldset.folds[j]]
            train_arr = _triples_array(foldset, train_pos)

            tf_train = TriplesFactory.from_labeled_triples(
                train_arr,
                entity_to_id=tf_global.entity_to_id,
                relation_to_id=tf_global.relation_to_id,
            )
            tf_test = TriplesFactory.from_labeled_triples(
                test_arr,
                entity_to_id=tf_global.entity_to_id,
                relation_to_id=tf_global.relation_to_id,
            )
            result = pipeline(
                training=tf_train,
                testing=tf_test,
                model=model_name,
                model_kwargs=dict(embedding_dim=embedding_dim),
                training_kwargs=dict(num_epochs=num_epochs, batch_size=batch_size),
                optimizer="Adam",
                optimizer_kwargs=dict(lr=lr),
                random_seed=seed,
                use_tqdm=False,
            )
            m = {
                "MRR": result.get_metric("both.realistic.inverse_harmonic_mean_rank"),
                "Hits@1": result.get_metric("both.realistic.hits_at_1"),
                "Hits@3": result.get_metric("both.realistic.hits_at_3"),
                "Hits@10": result.get_metric("both.realistic.hits_at_10"),
            }
            if compute_f1:
                m["F1"] = _f1_pairwise_kge(
                    result.model, test_arr, all_triples_set, tf_global, torch, seed=seed
                )
            fold_metrics.append(m)
            if verbose:
                line = "  ".join(f"{k}={m[k]:.4f}" for k in metric_keys)
                print(f"  Fold {i + 1}: {line}")

        per_fold[model_name] = fold_metrics
        avg[model_name] = {
            k: float(np.mean([fm[k] for fm in fold_metrics])) for k in metric_keys
        }
        std[model_name] = {
            k: float(np.std([fm[k] for fm in fold_metrics])) for k in metric_keys
        }
        if verbose:
            print("  -> Avg: " + "  ".join(f"{k}={avg[model_name][k]:.4f}" for k in metric_keys))

    return {
        "models": list(models),
        "metric_keys": metric_keys,
        "avg": avg,
        "std": std,
        "per_fold": per_fold,
        "k": foldset.k,
        "epochs": num_epochs,
        "dim": embedding_dim,
    }
