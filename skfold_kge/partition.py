"""Particionamiento k-fold estratificado (Round-Robin).

El algoritmo agrupa las filas por el valor de una columna de estrato
(``stratify_by``), baraja cada grupo de forma reproducible y reparte sus
elementos entre los ``k`` folds mediante Round-Robin. Así cada fold recibe
aproximadamente ``1/k`` de las filas de cada estrato, preservando la
distribución del conjunto original.

Casos de uso:

* **Grafos de conocimiento (KGE):** ``stratify_by`` = columna de relación.
  Cada fold mantiene la proporción de cada tipo de relación.
* **Clasificación / noticias falsas:** ``stratify_by`` = columna de clase
  (``label``). Cada fold mantiene la proporción de clases.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Centinela para agrupar todas las filas con estrato faltante (NaN) bajo una
# misma clave determinista, en lugar de descartarlas silenciosamente.
_NA_SENTINEL = "__NaN__"


def _stratum_key(value: object) -> object:
    """Devuelve una clave de estrato estable, mapeando NaN al centinela."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _NA_SENTINEL
    try:
        if pd.isna(value):
            return _NA_SENTINEL
    except (TypeError, ValueError):
        pass
    return value


class FoldSet:
    """Conjunto de ``k`` folds estratificados sobre un ``DataFrame``.

    Una instancia guarda el ``DataFrame`` limpio (tras deduplicación) y, por
    cada fold, los índices posicionales de las filas que le corresponden.
    Provee utilidades para iterar pares entrenamiento/prueba, exportar y
    verificar la integridad de la partición.

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Datos ya deduplicados, con índice ``RangeIndex`` (0..n-1).
    folds : list of list of int
        Índices posicionales de ``clean_df`` asignados a cada fold.
    stratify_by : str
        Nombre de la columna usada como estrato.
    k : int
        Número de folds.
    seed : int
        Semilla usada para el barajado (reproducibilidad).
    triple_columns : tuple of str, optional
        ``(sujeto, relación, objeto)`` cuando los datos son tripletas de un
        grafo de conocimiento. Habilita reportes de solapamiento de entidades.
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        folds: List[List[int]],
        stratify_by: str,
        k: int,
        seed: int,
        triple_columns: Optional[Tuple[str, str, str]] = None,
        n_input: Optional[int] = None,
        n_dedup_removed: int = 0,
        n_na_removed: int = 0,
    ) -> None:
        self.clean_df = clean_df
        self.folds = folds
        self.stratify_by = stratify_by
        self.k = k
        self.seed = seed
        self.triple_columns = triple_columns
        self.n_input = n_input if n_input is not None else len(clean_df)
        self.n_dedup_removed = n_dedup_removed
        self.n_na_removed = n_na_removed

    # ------------------------------------------------------------------ #
    # Acceso a los datos
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.k

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for i in range(self.k):
            yield self.fold_frame(i)

    def sizes(self) -> List[int]:
        """Número de filas de cada fold."""
        return [len(f) for f in self.folds]

    def fold_frame(self, i: int) -> pd.DataFrame:
        """``DataFrame`` (copia) con las filas del fold ``i`` (0-indexado)."""
        return self.clean_df.iloc[self.folds[i]].copy().reset_index(drop=True)

    def train_test(self, i: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Par ``(train, test)`` dejando el fold ``i`` como prueba."""
        test_idx = self.folds[i]
        train_idx = [p for j in range(self.k) if j != i for p in self.folds[j]]
        train = self.clean_df.iloc[train_idx].copy().reset_index(drop=True)
        test = self.clean_df.iloc[test_idx].copy().reset_index(drop=True)
        return train, test

    def iter_train_test(self) -> Iterator[Tuple[int, pd.DataFrame, pd.DataFrame]]:
        """Itera ``(i, train, test)`` para cada fold de prueba."""
        for i in range(self.k):
            train, test = self.train_test(i)
            yield i, train, test

    def labels(self) -> pd.Series:
        """Serie con el valor de estrato de cada fila de ``clean_df``."""
        return self.clean_df[self.stratify_by]

    # ------------------------------------------------------------------ #
    # Verificación y reportes
    # ------------------------------------------------------------------ #
    def verify(self):
        """Calcula el :class:`~skfold_kge.verify.IntegrityReport` de la partición."""
        from .verify import build_integrity_report

        return build_integrity_report(self)

    # Alias semántico
    report = verify

    # ------------------------------------------------------------------ #
    # Exportación
    # ------------------------------------------------------------------ #
    def _frame_with_label(self, i: int, label_col: str) -> pd.DataFrame:
        frame = self.fold_frame(i)
        if label_col not in frame.columns:
            frame[label_col] = self.clean_df[self.stratify_by].iloc[self.folds[i]].values
        return frame

    def to_excel(
        self,
        path: str,
        label_col: str = "label",
        rename: Optional[dict] = None,
    ) -> str:
        """Exporta cada fold a una hoja del archivo Excel ``path``."""
        from .io import export_folds_excel

        return export_folds_excel(self, path, label_col=label_col, rename=rename)

    def to_csv_dir(
        self,
        directory: str,
        label_col: str = "label",
        rename: Optional[dict] = None,
    ) -> List[str]:
        """Exporta cada fold como un CSV dentro de ``directory``."""
        from .io import export_folds_csv_dir

        return export_folds_csv_dir(self, directory, label_col=label_col, rename=rename)


class StratifiedPartitioner:
    """Particionador k-fold estratificado por una columna categórica.

    Parameters
    ----------
    k : int, default 5
        Número de folds. Con ``k=5`` cada fold de prueba contiene ~20% de los
        datos (proporción estándar en benchmarks KGE como FB15k-237 / WN18RR).
    stratify_by : str
        Nombre de la columna usada como estrato (p. ej. la relación en un KG o
        la clase en clasificación).
    seed : int, default 42
        Semilla del barajado para resultados reproducibles.
    dedup : bool, default True
        Si ``True`` elimina filas duplicadas (todas las columnas) antes de
        particionar. Tripletas repetidas sesgarían la distribución.
    dropna_stratum : bool, default False
        Si ``True`` descarta filas cuyo estrato es NaN. Si ``False`` (por
        defecto) las conserva agrupadas en un estrato propio y las reporta,
        para no ocultar problemas de calidad del dato.
    triple_columns : tuple of str, optional
        ``(sujeto, relación, objeto)``. Si se omite y el ``DataFrame`` tiene
        exactamente 3 columnas, se infieren automáticamente.

    Examples
    --------
    >>> part = StratifiedPartitioner(k=5, stratify_by="Column2", seed=42)
    >>> folds = part.fit_transform(df)
    >>> folds.sizes()
    [640, 639, 635, 632, 630]
    """

    def __init__(
        self,
        k: int = 5,
        stratify_by: Optional[str] = None,
        seed: int = 42,
        dedup: bool = True,
        dropna_stratum: bool = False,
        triple_columns: Optional[Tuple[str, str, str]] = None,
        method: str = "round_robin",
    ) -> None:
        if k < 2:
            raise ValueError(f"k debe ser >= 2, se recibió {k}.")
        if stratify_by is None:
            raise ValueError("Debe indicar 'stratify_by' (columna de estrato).")
        if method != "round_robin":
            raise ValueError(f"Método no soportado: {method!r}. Use 'round_robin'.")
        self.k = k
        self.stratify_by = stratify_by
        self.seed = seed
        self.dedup = dedup
        self.dropna_stratum = dropna_stratum
        self.triple_columns = triple_columns
        self.method = method

    def fit_transform(self, df: pd.DataFrame) -> FoldSet:
        """Particiona ``df`` y devuelve un :class:`FoldSet`.

        Parameters
        ----------
        df : pandas.DataFrame
            Datos a particionar. Debe contener la columna ``stratify_by``.

        Returns
        -------
        FoldSet
        """
        if self.stratify_by not in df.columns:
            raise KeyError(
                f"La columna de estrato {self.stratify_by!r} no está en el "
                f"DataFrame. Columnas: {list(df.columns)}"
            )

        n_input = len(df)
        clean = df
        n_dedup_removed = 0
        if self.dedup:
            clean = clean.drop_duplicates()
            n_dedup_removed = n_input - len(clean)
        n_na_removed = 0
        if self.dropna_stratum:
            before = len(clean)
            clean = clean[clean[self.stratify_by].notna()]
            n_na_removed = before - len(clean)
        clean = clean.reset_index(drop=True)

        triple_columns = self._resolve_triple_columns(clean)

        # Agrupar índices posicionales por estrato, en orden de primera aparición.
        groups: dict = {}
        order: List[object] = []
        for pos, value in enumerate(clean[self.stratify_by].to_numpy()):
            key = _stratum_key(value)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(pos)

        # Barajar cada grupo y repartir Round-Robin entre los k folds.
        rng = np.random.RandomState(self.seed)
        folds: List[List[int]] = [[] for _ in range(self.k)]
        for key in order:
            idxs = np.array(groups[key], dtype=int)
            rng.shuffle(idxs)
            for j, pos in enumerate(idxs):
                folds[j % self.k].append(int(pos))

        return FoldSet(
            clean_df=clean,
            folds=folds,
            stratify_by=self.stratify_by,
            k=self.k,
            seed=self.seed,
            triple_columns=triple_columns,
            n_input=n_input,
            n_dedup_removed=n_dedup_removed,
            n_na_removed=n_na_removed,
        )

    def _resolve_triple_columns(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[str, str, str]]:
        if self.triple_columns is not None:
            missing = [c for c in self.triple_columns if c not in df.columns]
            if missing:
                raise KeyError(f"triple_columns ausentes en el DataFrame: {missing}")
            return tuple(self.triple_columns)  # type: ignore[return-value]
        if df.shape[1] == 3:
            return tuple(df.columns[:3])  # type: ignore[return-value]
        return None


def partition(
    df: pd.DataFrame,
    k: int = 5,
    stratify_by: Optional[str] = None,
    seed: int = 42,
    dedup: bool = True,
    dropna_stratum: bool = False,
    triple_columns: Optional[Tuple[str, str, str]] = None,
) -> FoldSet:
    """Atajo funcional equivalente a ``StratifiedPartitioner(...).fit_transform(df)``."""
    return StratifiedPartitioner(
        k=k,
        stratify_by=stratify_by,
        seed=seed,
        dedup=dedup,
        dropna_stratum=dropna_stratum,
        triple_columns=triple_columns,
    ).fit_transform(df)
