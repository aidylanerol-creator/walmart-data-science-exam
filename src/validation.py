"""Esquema de validación temporal con ventanas rodantes.

Nunca se usa split aleatorio: el objetivo es simular el uso real del modelo,
donde se entrena con el pasado y se predice el futuro.
"""
from __future__ import annotations

import pandas as pd


def rolling_origin_splits(
    dates: pd.Series,
    n_folds: int = 4,
    test_days: int = 14,
    gap_days: int = 0,
):
    """Genera folds de validación con origen rodante.

    Cada fold entrena con todo el histórico previo y evalúa una ventana de
    `test_days` días. Los folds se anclan al final del periodo, retrocediendo
    `test_days` por fold. Yields: (train_mask, test_mask, test_start, test_end).
    """
    dates = pd.to_datetime(dates)
    max_date = dates.max()

    for fold in range(n_folds, 0, -1):
        test_end = max_date - pd.Timedelta(days=test_days * (fold - 1))
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        train_end = test_start - pd.Timedelta(days=1 + gap_days)

        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)
        yield train_mask, test_mask, test_start, test_end
