import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def label_entities_by_relation(subject, relation, obj):
    """Clasifica (sujeto, objeto) según el tipo semántico de la relación. Conservada para referencia."""
    rel_map = {
        "ALLIED_WITH": ("Casa", "Casa"),
        "BRANCH_OF":   ("Casa", "Casa"),
        "FOUNDED_BY":  ("Casa", "Persona"),
        "HEIR_TO":     ("Persona", "Persona"),
        "IN_REGION":   ("Casa", "Región"),
        "LED_BY":      ("Casa", "Persona"),
        "PARENT_OF":   ("Persona", "Persona"),
        "SEAT_OF":     ("Seat", "Casa"),
        "SPOUSE":      ("Persona", "Persona"),
        "SWORN_TO":    ("Casa", "Casa"),
    }
    if relation in rel_map:
        return rel_map[relation]

    def heuristic_label(entity):
        if not isinstance(entity, str):
            return "Unknown"
        e = entity.strip()
        if e.startswith("House"):
            return "Casa"
        parts = e.split()
        if len(parts) == 0:
            return "Unknown"
        if parts[0].lower() == "the":
            return "Región"
        if len(parts) <= 3:
            all_cap = all(p and p[0].isupper() for p in parts)
            if all_cap:
                return "Persona"
        return "Unknown"

    return heuristic_label(subject), heuristic_label(obj)


def triple_label_for_stratification(subject, relation, obj):
    """Etiqueta de estrato: tipo de relación de la tripleta."""
    return relation


def create_stratified_partitions_by_triple(triples_df, k_folds=5, random_seed=42):
    np.random.seed(random_seed)
    triples = list(triples_df.itertuples(index=False, name=None))

    grouped       = defaultdict(list)
    triple_labels = {}

    for s, r, o in triples:
        lbl = triple_label_for_stratification(s, r, o)
        grouped[lbl].append((s, r, o))
        triple_labels[(s, r, o)] = lbl

    folds = [[] for _ in range(k_folds)]
    for lbl, tlist in grouped.items():
        tcopy = tlist.copy()
        np.random.shuffle(tcopy)
        for idx, triple in enumerate(tcopy):
            folds[idx % k_folds].append(triple)

    return folds, triple_labels


def fold_summary_and_checks(folds, triple_labels):
    summaries          = []
    fold_entities_sets = []
    fold_objects_sets  = []

    for i, fold in enumerate(folds):
        lbl_counts    = Counter([triple_labels[t] for t in fold])
        objects       = [o for (_, _, o) in fold]
        obj_counter   = Counter(objects)
        repeated_objs = [obj for obj, cnt in obj_counter.items() if cnt > 1]
        relations     = [r for (_, r, _) in fold]
        rel_counter   = Counter(relations)
        subjects      = [s for (s, _, _) in fold]
        entities      = set(subjects) | set(objects)

        summaries.append({
            "fold":                   i + 1,
            "by_label":               dict(lbl_counts),
            "repeated_objects_count": len(repeated_objs),
            "some_repeated_objects":  repeated_objs[:8],
            "unique_relations":       len(rel_counter),
            "relation_freq_sample":   list(rel_counter.items())[:8],
        })
        fold_entities_sets.append(entities)
        fold_objects_sets.append(set(objects))

    common_entities = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            commons_ent = fold_entities_sets[i] & fold_entities_sets[j]
            common_entities.append({
                "pair":         f"{i+1}-{j+1}",
                "common_count": len(commons_ent),
                "sample":       list(commons_ent)[:5],
            })

    seen     = {}
    overlaps = []
    for i, fold in enumerate(folds):
        for t in fold:
            if t in seen:
                overlaps.append({"triple": t, "first_fold": seen[t] + 1, "second_fold": i + 1})
            else:
                seen[t] = i

    return summaries, common_entities, overlaps


def compute_filtered_ranks(test_triples, entities_list, score_fn, corrupt_side, filter_triples):
    ranks = []
    for s, r, o in test_triples:
        true_score       = score_fn(s, r, o)
        candidate_scores = []

        for entity in entities_list:
            if corrupt_side == "subject":
                corrupted = (entity, r, o)
            elif corrupt_side == "object":
                corrupted = (s, r, entity)
            else:
                continue

            if corrupted not in filter_triples:
                candidate_scores.append(score_fn(*corrupted))

        all_scores = sorted(candidate_scores + [true_score], reverse=True)
        ranks.append(all_scores.index(true_score) + 1)

    return ranks


def compute_metrics_from_ranks(ranks):
    if not ranks:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    mrr = hits1 = hits3 = hits10 = 0
    for rank in ranks:
        mrr += 1.0 / rank
        if rank <= 1:  hits1  += 1
        if rank <= 3:  hits3  += 1
        if rank <= 10: hits10 += 1

    n = len(ranks)
    return {
        "MRR":     mrr    / n,
        "Hits@1":  hits1  / n,
        "Hits@3":  hits3  / n,
        "Hits@10": hits10 / n,
    }


def stratified_cv_fake_news(df_news, text_col="text", label_col="label", k=5, seed=42):
    X   = df_news[text_col].values
    y   = df_news[label_col].values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vec  = TfidfVectorizer(max_features=50_000, sublinear_tf=True, ngram_range=(1, 2))
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)

        clf   = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y_train)
        y_hat = clf.predict(X_te)

        fold_results.append({
            "Fold":      fold_idx + 1,
            "F1":        f1_score(y_test, y_hat, average="macro"),
            "Precision": precision_score(y_test, y_hat, average="macro", zero_division=0),
            "Recall":    recall_score(y_test, y_hat, average="macro", zero_division=0),
            "Accuracy":  float((y_hat == y_test).mean()),
        })

    return pd.DataFrame(fold_results)
