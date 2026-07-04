"""Construcción del dataset de features para el pronóstico de demanda.

Principio central: para predecir el día t con horizonte h, solo se usa
información disponible hasta t - h. Todos los lags y ventanas móviles se
desplazan al menos `horizon` días. `replenishment_signal` se excluye
explícitamente por leakage (ver notebooks/01_eda.ipynb, sección 10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["store_id", "category"]
TARGET = "units_sold"

# Columnas que NUNCA entran como features:
# - replenishment_signal: leakage (ventana móvil que incluye el día corriente)
# - métricas del mismo día (amount_*, *_transactions, avg_ticket): no
#   disponibles al momento de predecir
LEAKY_COLS = [
    "replenishment_signal", "amount_total", "amount_cash", "amount_card",
    "total_transactions", "cash_transactions", "card_transactions", "avg_ticket",
]


def load_data(data_dir: str = "data"):
    """Carga los tres archivos fuente con fechas parseadas."""
    transactions = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    calendar = pd.read_csv(f"{data_dir}/calendar.csv", parse_dates=["date"])
    return transactions, stores, calendar


def build_panel(transactions: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Reindexa a un panel completo fecha x tienda x categoría.

    Las 42 combinaciones ausentes (huecos de 2-3 días en 3 tiendas) entran
    como filas con target nulo; así los lags mantienen alineación temporal.
    """
    full_index = pd.MultiIndex.from_product(
        [calendar["date"].unique(),
         sorted(transactions["store_id"].unique()),
         sorted(transactions["category"].unique())],
        names=["date", "store_id", "category"],
    )
    panel = (
        transactions.set_index(["date", "store_id", "category"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["store_id", "category", "date"])
        .reset_index(drop=True)
    )
    return panel


def impute_target_for_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Crea `units_filled`: target interpolado linealmente por serie.

    Se usa SOLO para calcular lags/ventanas (evita propagar nulos);
    las filas con target original nulo se excluyen del entrenamiento.
    """
    panel = panel.copy()
    panel["units_filled"] = (
        panel.groupby(KEY)[TARGET]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    return panel


def add_lag_features(panel: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """Lags y ventanas móviles del target, desplazados >= horizon días.

    Con horizon=14: lags de 14, 21 y 28 días (mismo día de la semana) y
    medias/desviaciones móviles de 7 y 28 días calculadas hasta t-14.
    """
    panel = panel.copy()
    g = panel.groupby(KEY)["units_filled"]

    for lag in [horizon, horizon + 7, horizon + 14]:
        panel[f"lag_{lag}"] = g.shift(lag)

    shifted = g.shift(horizon)  # serie hasta t - horizon
    for window in [7, 28]:
        roll = shifted.groupby([panel[k] for k in KEY]).rolling(window, min_periods=window // 2)
        panel[f"rmean_{window}_h{horizon}"] = roll.mean().reset_index(drop=True)
        panel[f"rstd_{window}_h{horizon}"] = roll.std().reset_index(drop=True)

    return panel


def add_calendar_features(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Une variables de calendario (conocidas de antemano: sin leakage)."""
    cal_cols = [
        "date", "day_of_week", "week_of_year", "month", "quarter",
        "is_holiday", "is_payday", "is_weekend",
        "is_navidad_season", "is_buen_fin", "is_semana_santa",
    ]
    panel = panel.merge(calendar[cal_cols], on="date", how="left")
    bool_cols = [c for c in cal_cols if c.startswith("is_")]
    panel[bool_cols] = panel[bool_cols].astype(int)
    return panel


def add_store_features(panel: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Une atributos estáticos de tienda (categóricos como dtype category)."""
    panel = panel.merge(stores, on="store_id", how="left")
    panel["store_age_years"] = panel["date"].dt.year - panel["opening_year"]
    for col in ["store_format", "region", "socioeconomic_level", "category"]:
        panel[col] = panel[col].astype("category")
    panel[["has_pharmacy", "has_fuel_station"]] = panel[["has_pharmacy", "has_fuel_station"]].astype(int)
    return panel


def build_dataset(data_dir: str = "data", horizon: int = 14) -> tuple[pd.DataFrame, list[str]]:
    """Pipeline completo. Devuelve (dataset, lista_de_features).

    Nota: `has_promotion` del día a predecir se asume conocida de antemano
    (las promociones se planean); es la única variable operativa que se usa.
    """
    transactions, stores, calendar = load_data(data_dir)

    panel = build_panel(transactions, calendar)
    panel = impute_target_for_lags(panel)
    panel = add_lag_features(panel, horizon=horizon)
    panel = add_calendar_features(panel, calendar)
    panel = add_store_features(panel, stores)

    feature_cols = [
        c for c in panel.columns
        if c not in {"date", "store_id", TARGET, "units_filled", "opening_year", *LEAKY_COLS}
    ]
    return panel, feature_cols
