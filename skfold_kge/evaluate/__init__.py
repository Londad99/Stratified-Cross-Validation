"""Extras de evaluación (dependencias opcionales).

* :func:`cross_validate_kge` — requiere ``pip install skfold-kge[kge]``
  (pykeen + torch). Compara modelos KGE bajo la partición estratificada.
* :func:`cross_validate_text` — requiere ``pip install skfold-kge[text]``
  (scikit-learn). Clasificación de texto / noticias falsas con F1 macro.

Las dependencias pesadas se importan de forma perezosa dentro de cada función,
por lo que ``import skfold_kge`` permanece ligero.
"""

from __future__ import annotations

from .classification import (
    cross_validate_text,
    load_isot,
    load_liar,
    load_welfake,
)
from .kge import cross_validate_kge

__all__ = [
    "cross_validate_kge",
    "cross_validate_text",
    "load_isot",
    "load_liar",
    "load_welfake",
]
