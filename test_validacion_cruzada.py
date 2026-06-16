import numpy as np
import pandas as pd
import pytest

from validacion_cruzada import (
    label_entities_by_relation,
    triple_label_for_stratification,
    create_stratified_partitions_by_triple,
    fold_summary_and_checks,
    compute_filtered_ranks,
    compute_metrics_from_ranks,
    stratified_cv_fake_news,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """Dataset mínimo: 3 relaciones × 3 tripletas únicas cada una."""
    data = [
        ("A1", "ALLIED_WITH", "B1"),
        ("A2", "ALLIED_WITH", "B2"),
        ("A3", "ALLIED_WITH", "B3"),
        ("C1", "LED_BY",      "D1"),
        ("C2", "LED_BY",      "D2"),
        ("C3", "LED_BY",      "D3"),
        ("E1", "SPOUSE",      "F1"),
        ("E2", "SPOUSE",      "F2"),
        ("E3", "SPOUSE",      "F3"),
    ]
    return pd.DataFrame(data, columns=["Column1", "Column2", "Column3"])


@pytest.fixture
def folds_and_labels(small_df):
    return create_stratified_partitions_by_triple(small_df, k_folds=3, random_seed=0)


@pytest.fixture
def df_news():
    """Dataset balanceado de noticias para pruebas de fake news."""
    np.random.seed(0)
    texts  = [f"texto {i} {'falsa' if i < 100 else 'real'}" for i in range(200)]
    labels = [0] * 100 + [1] * 100
    return pd.DataFrame({"text": texts, "label": labels})


# ── triple_label_for_stratification ──────────────────────────────────────────

class TestTripleLabelForStratification:
    def test_devuelve_la_relacion(self):
        assert triple_label_for_stratification("A", "ALLIED_WITH", "B") == "ALLIED_WITH"

    def test_ignora_sujeto_y_objeto(self):
        assert triple_label_for_stratification("X", "SPOUSE", "Y") == "SPOUSE"

    def test_relacion_desconocida(self):
        assert triple_label_for_stratification("P", "CUSTOM_REL", "Q") == "CUSTOM_REL"


# ── label_entities_by_relation ────────────────────────────────────────────────

class TestLabelEntitiesByRelation:
    def test_relacion_conocida_allied_with(self):
        assert label_entities_by_relation("A", "ALLIED_WITH", "B") == ("Casa", "Casa")

    def test_relacion_founded_by(self):
        assert label_entities_by_relation("X", "FOUNDED_BY", "Y") == ("Casa", "Persona")

    def test_relacion_led_by(self):
        assert label_entities_by_relation("X", "LED_BY", "Y") == ("Casa", "Persona")

    def test_heuristica_house(self):
        subj_lbl, _ = label_entities_by_relation("House Stark", "UNKNOWN", "Jon Snow")
        assert subj_lbl == "Casa"

    def test_heuristica_the_region(self):
        _, obj_lbl = label_entities_by_relation("X", "UNKNOWN", "The North")
        assert obj_lbl == "Región"

    def test_heuristica_persona_nombre_compuesto(self):
        _, obj_lbl = label_entities_by_relation("X", "UNKNOWN", "Jon Snow")
        assert obj_lbl == "Persona"


# ── create_stratified_partitions_by_triple ────────────────────────────────────

class TestCreateStratifiedPartitions:
    def test_numero_de_folds(self, small_df):
        folds, _ = create_stratified_partitions_by_triple(small_df, k_folds=3)
        assert len(folds) == 3

    def test_total_tripletas_conservadas(self, small_df):
        folds, _ = create_stratified_partitions_by_triple(small_df, k_folds=3)
        assert sum(len(f) for f in folds) == len(small_df)

    def test_sin_overlap_entre_folds(self, small_df):
        folds, _ = create_stratified_partitions_by_triple(small_df, k_folds=3)
        seen = set()
        for fold in folds:
            for t in fold:
                assert t not in seen, f"Tripleta duplicada: {t}"
                seen.add(t)

    def test_distribucion_estratificada(self, small_df):
        """Con 3 tripletas por relación y 3 folds, cada fold recibe 1 por relación."""
        folds, _ = create_stratified_partitions_by_triple(small_df, k_folds=3)
        for fold in folds:
            relaciones = [t[1] for t in fold]
            assert relaciones.count("ALLIED_WITH") == 1
            assert relaciones.count("LED_BY") == 1
            assert relaciones.count("SPOUSE") == 1

    def test_reproducibilidad_con_seed(self, small_df):
        folds1, _ = create_stratified_partitions_by_triple(small_df, k_folds=3, random_seed=7)
        folds2, _ = create_stratified_partitions_by_triple(small_df, k_folds=3, random_seed=7)
        assert folds1 == folds2

    def test_seeds_distintas_producen_resultados_distintos(self, small_df):
        folds1, _ = create_stratified_partitions_by_triple(small_df, k_folds=3, random_seed=1)
        folds2, _ = create_stratified_partitions_by_triple(small_df, k_folds=3, random_seed=99)
        assert folds1 != folds2

    def test_triple_labels_cubre_todas_las_tripletas(self, small_df):
        folds, labels = create_stratified_partitions_by_triple(small_df, k_folds=3)
        for fold in folds:
            for t in fold:
                assert t in labels


# ── compute_metrics_from_ranks ────────────────────────────────────────────────

class TestComputeMetricsFromRanks:
    def test_ranking_perfecto(self):
        m = compute_metrics_from_ranks([1, 1, 1])
        assert m["MRR"]    == pytest.approx(1.0)
        assert m["Hits@1"] == pytest.approx(1.0)
        assert m["Hits@3"] == pytest.approx(1.0)

    def test_rango_dos(self):
        m = compute_metrics_from_ranks([2])
        assert m["MRR"]    == pytest.approx(0.5)
        assert m["Hits@1"] == pytest.approx(0.0)
        assert m["Hits@3"] == pytest.approx(1.0)

    def test_lista_vacia_devuelve_ceros(self):
        m = compute_metrics_from_ranks([])
        assert m == {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    def test_hits_at_10_parcial(self):
        m = compute_metrics_from_ranks([10, 11])
        assert m["Hits@10"] == pytest.approx(0.5)

    def test_mrr_es_promedio_de_reciprocos(self):
        m = compute_metrics_from_ranks([1, 2])
        assert m["MRR"] == pytest.approx((1.0 + 0.5) / 2)

    def test_rango_alto_no_cuenta_en_hits(self):
        m = compute_metrics_from_ranks([100])
        assert m["Hits@1"]  == 0.0
        assert m["Hits@3"]  == 0.0
        assert m["Hits@10"] == 0.0


# ── compute_filtered_ranks ────────────────────────────────────────────────────

class TestComputeFilteredRanks:
    def test_rango_uno_cuando_score_maximo(self):
        score_fn = lambda s, r, o: 1.0 if o == "B" else 0.0
        ranks = compute_filtered_ranks(
            [("X", "R", "B")], ["A", "B", "C"],
            score_fn, corrupt_side="object", filter_triples=set()
        )
        assert ranks == [1]

    def test_rango_correcto_cuando_hay_candidatos_mejores(self):
        # C tiene score 0; B tiene score 1 y compite → C queda en rango 2
        score_fn = lambda s, r, o: 1.0 if o == "B" else 0.0
        ranks = compute_filtered_ranks(
            [("X", "R", "C")], ["A", "B", "C"],
            score_fn, corrupt_side="object", filter_triples=set()
        )
        assert ranks == [2]

    def test_corrupt_side_subject(self):
        score_fn = lambda s, r, o: 1.0 if s == "A" else 0.0
        ranks = compute_filtered_ranks(
            [("A", "R", "Z")], ["A", "B", "C"],
            score_fn, corrupt_side="subject", filter_triples=set()
        )
        assert ranks == [1]

    def test_filtrado_excluye_competidores(self):
        """Al filtrar B, deja de competir y C sube al rango 1."""
        score_fn    = lambda s, r, o: 1.0 if o == "B" else 0.0
        filter_set  = {("X", "R", "B")}
        ranks = compute_filtered_ranks(
            [("X", "R", "C")], ["A", "B", "C"],
            score_fn, corrupt_side="object", filter_triples=filter_set
        )
        assert ranks == [1]

    def test_multiples_tripletas(self):
        score_fn = lambda s, r, o: {"A": 3.0, "B": 2.0, "C": 1.0}.get(o, 0.0)
        ranks = compute_filtered_ranks(
            [("X", "R", "A"), ("X", "R", "C")], ["A", "B", "C"],
            score_fn, corrupt_side="object", filter_triples=set()
        )
        assert ranks[0] == 1  # A tiene el score más alto
        assert ranks[1] == 3  # C tiene el score más bajo


# ── fold_summary_and_checks ───────────────────────────────────────────────────

class TestFoldSummaryAndChecks:
    def test_sin_overlap(self, folds_and_labels):
        folds, labels = folds_and_labels
        _, _, overlaps = fold_summary_and_checks(folds, labels)
        assert overlaps == []

    def test_numero_de_summaries_igual_a_folds(self, folds_and_labels):
        folds, labels = folds_and_labels
        summaries, _, _ = fold_summary_and_checks(folds, labels)
        assert len(summaries) == len(folds)

    def test_summaries_tienen_campos_requeridos(self, folds_and_labels):
        folds, labels = folds_and_labels
        summaries, _, _ = fold_summary_and_checks(folds, labels)
        for s in summaries:
            assert "fold" in s
            assert "by_label" in s
            assert "unique_relations" in s

    def test_detecta_overlap_artificial(self):
        """Una tripleta presente en dos folds debe aparecer en overlaps."""
        triple = ("A", "R", "B")
        folds  = [[triple], [triple]]
        labels = {triple: "R"}
        _, _, overlaps = fold_summary_and_checks(folds, labels)
        assert len(overlaps) == 1
        assert overlaps[0]["triple"] == triple

    def test_pares_de_entidades_comunes(self, folds_and_labels):
        folds, labels = folds_and_labels
        _, common, _ = fold_summary_and_checks(folds, labels)
        # Con 3 folds hay C(3,2)=3 pares
        assert len(common) == 3


# ── stratified_cv_fake_news ───────────────────────────────────────────────────

class TestStratifiedCvFakeNews:
    def test_devuelve_dataframe_con_k_filas(self, df_news):
        result = stratified_cv_fake_news(df_news, k=3)
        assert len(result) == 3

    def test_columnas_presentes(self, df_news):
        result = stratified_cv_fake_news(df_news, k=3)
        assert set(result.columns) >= {"Fold", "F1", "Precision", "Recall", "Accuracy"}

    def test_metricas_entre_0_y_1(self, df_news):
        result = stratified_cv_fake_news(df_news, k=3)
        for col in ["F1", "Precision", "Recall", "Accuracy"]:
            assert result[col].between(0, 1).all(), f"{col} fuera de rango [0,1]"

    def test_columna_fold_es_consecutiva(self, df_news):
        result = stratified_cv_fake_news(df_news, k=3)
        assert list(result["Fold"]) == [1, 2, 3]

    def test_reproducibilidad(self, df_news):
        r1 = stratified_cv_fake_news(df_news, k=3, seed=42)
        r2 = stratified_cv_fake_news(df_news, k=3, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_seed_distinta_puede_dar_resultado_distinto(self, df_news):
        r1 = stratified_cv_fake_news(df_news, k=3, seed=0)
        r2 = stratified_cv_fake_news(df_news, k=3, seed=999)
        # No necesariamente distintos, pero al menos no falla
        assert len(r1) == len(r2)
