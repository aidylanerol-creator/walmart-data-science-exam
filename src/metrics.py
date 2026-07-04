"""Métricas de evaluación del pronóstico."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(y_true, y_pred) -> float:
    """Weighted MAPE: sum(|error|) / sum(|real|).

    Métrica estándar en retail: robusta a días de venta baja (a diferencia
    del MAPE) y interpretable como % de error sobre la demanda total.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).mean())


def bias(y_true, y_pred) -> float:
    """Sesgo relativo: >0 sobre-pronostica (sobre-stock), <0 sub-pronostica (quiebres)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float((y_pred - y_true).sum() / np.abs(y_true).sum())


def evaluate(df: pd.DataFrame, y_col: str, pred_col: str, by=None) -> pd.DataFrame:
    """Tabla de métricas, opcionalmente desagregada por `by` (ej. category)."""
    def _calc(g):
        return pd.Series({
            "wmape": wmape(g[y_col], g[pred_col]),
            "mae": mae(g[y_col], g[pred_col]),
            "bias": bias(g[y_col], g[pred_col]),
            "n": len(g),
        })

    if by is None:
        return _calc(df).to_frame("global").T
    return df.groupby(by, observed=True).apply(_calc)
