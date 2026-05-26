"""
Geração de Dataset Sintético — Monitoramento de Desmatamento e Queimadas via Satélites
Mínimo: 1.000 linhas e 10 colunas
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N = 2000

# ── Coordenadas no bioma Amazônia/Cerrado (Brasil) ──────────────────────────
lat  = np.random.uniform(-15.0, 5.0,  N)   # latitude
lon  = np.random.uniform(-73.0, -45.0, N)  # longitude

# ── Série temporal ────────────────────────────────────────────────────────────
start_date = datetime(2018, 1, 1)
dates = [start_date + timedelta(days=int(d)) for d in np.random.uniform(0, 365*5, N)]
month = np.array([d.month for d in dates])

# ── Fatores ambientais base ───────────────────────────────────────────────────
# Temperatura superficial (°C) — mais alta na época seca (jun-set)
dry_season = ((month >= 6) & (month <= 9)).astype(float)
temp_surface = 28 + 6 * dry_season + np.random.normal(0, 2, N)

# Umidade relativa (%) — inversamente proporcional à temp
humidity = 75 - 20 * dry_season + np.random.normal(0, 5, N)
humidity = np.clip(humidity, 20, 100)

# NDVI — índice de vegetação (-1 a 1). Reduz com desmatamento e época seca
ndvi_base = 0.65 - 0.15 * dry_season + np.random.normal(0, 0.05, N)

# Precipitação acumulada 30 dias (mm)
precipitation = np.maximum(0, 150 - 120 * dry_season + np.random.normal(0, 30, N))

# Vento médio (km/h)
wind_speed = np.abs(np.random.normal(12, 4, N))

# NDWI — índice hídrico (-1 a 1)
ndwi = 0.3 - 0.2 * dry_season + 0.4 * (precipitation / 200) + np.random.normal(0, 0.05, N)

# FRP — Fire Radiative Power (MW), proxy de intensidade de fogo
frp_base = np.maximum(0, 5 * dry_season + np.random.exponential(2, N))

# Distância à estrada mais próxima (km)
dist_road = np.abs(np.random.exponential(25, N))

# Distância a área protegida (km)
dist_protected = np.abs(np.random.exponential(40, N))

# Densidade populacional (hab/km²)
pop_density = np.abs(np.random.exponential(10, N))

# Altitude (m)
altitude = np.abs(np.random.normal(300, 150, N))

# Dias sem chuva consecutivos
days_no_rain = np.maximum(0, 30 * dry_season + np.random.exponential(10, N) - precipitation / 10)

# ── Variável-alvo: risco_desmatamento (0=Baixo, 1=Médio, 2=Alto) ─────────────
risk_score = (
    0.30 * (temp_surface - 28) / 6
    + 0.20 * (1 - humidity / 100)
    + 0.25 * (1 - np.clip(ndvi_base, -1, 1))
    + 0.15 * (frp_base / 15)
    + 0.10 * (1 - np.clip(dist_road / 100, 0, 1))
    - 0.10 * np.clip(dist_protected / 100, 0, 1)
    + np.random.normal(0, 0.08, N)
)

risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

risco_label = pd.cut(
    risk_score,
    bins=[0, 0.35, 0.65, 1.0],
    labels=["Baixo", "Médio", "Alto"],
    include_lowest=True,
)

# ── Montar DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    "data":               [d.strftime("%Y-%m-%d") for d in dates],
    "latitude":           lat.round(4),
    "longitude":          lon.round(4),
    "mes":                month,
    "temp_superficie":    temp_surface.round(2),
    "umidade_relativa":   humidity.round(2),
    "ndvi":               ndvi_base.round(4),
    "ndwi":               ndwi.round(4),
    "precipitacao_30d":   precipitation.round(2),
    "velocidade_vento":   wind_speed.round(2),
    "frp_fogo":           frp_base.round(2),
    "dist_estrada_km":    dist_road.round(2),
    "dist_area_protegida_km": dist_protected.round(2),
    "densidade_pop":      pop_density.round(2),
    "altitude_m":         altitude.round(1),
    "dias_sem_chuva":     days_no_rain.round(1),
    "estacao_seca":       dry_season.astype(int),
    "risco_score":        risk_score.round(4),
    "risco_desmatamento": risco_label,
})

df.to_csv("dataset_desmatamento.csv", index=False)
print(f"Dataset gerado: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(df["risco_desmatamento"].value_counts())
print(df.head())
