"""Entrada/salida: carga de datasets y exportación de folds.

Reemplaza la lógica de Colab del notebook (``files.download``) por escritura a
disco estándar, válida en cualquier entorno.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .partition import FoldSet


def load_triples(
    path: str,
    sep: str = ";",
    columns: Optional[Sequence[str]] = None,
    header: object = "infer",
) -> pd.DataFrame:
    """Carga un CSV de tripletas/filas en un ``DataFrame``.

    Parameters
    ----------
    path : str
        Ruta o URL del CSV.
    sep : str, default ";"
        Separador de columnas.
    columns : sequence of str, optional
        Si se da, renombra las columnas leídas a estos nombres.
    header : {'infer', None, int}, default 'infer'
        Igual que en :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path, sep=sep, header=header)
    if columns is not None:
        if len(columns) != df.shape[1]:
            raise ValueError(
                f"Se pasaron {len(columns)} nombres pero el CSV tiene "
                f"{df.shape[1]} columnas."
            )
        df.columns = list(columns)
    return df


def _fold_frame_for_export(
    foldset: "FoldSet",
    i: int,
    label_col: str,
    rename: Optional[Dict[str, str]],
) -> pd.DataFrame:
    frame = foldset.clean_df.iloc[foldset.folds[i]].copy().reset_index(drop=True)
    # Añade la etiqueta de estrato como columna explícita.
    frame[label_col] = foldset.clean_df[foldset.stratify_by].iloc[foldset.folds[i]].values
    if rename:
        frame = frame.rename(columns=rename)
    return frame


def export_folds_excel(
    foldset: "FoldSet",
    path: str,
    label_col: str = "label",
    rename: Optional[Dict[str, str]] = None,
) -> str:
    """Escribe cada fold en una hoja ``Fold_i`` de un único archivo ``.xlsx``.

    Returns
    -------
    str
        La ruta escrita.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for i in range(foldset.k):
            frame = _fold_frame_for_export(foldset, i, label_col, rename)
            frame.to_excel(writer, sheet_name=f"Fold_{i + 1}", index=False)
    return path


def export_folds_csv_dir(
    foldset: "FoldSet",
    directory: str,
    label_col: str = "label",
    rename: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Escribe un CSV por fold (``Fold_1.csv`` …) dentro de ``directory``.

    Returns
    -------
    list of str
        Rutas de los archivos escritos.
    """
    os.makedirs(directory, exist_ok=True)
    paths: List[str] = []
    for i in range(foldset.k):
        frame = _fold_frame_for_export(foldset, i, label_col, rename)
        out = os.path.join(directory, f"Fold_{i + 1}.csv")
        frame.to_csv(out, index=False)
        paths.append(out)
    return paths
