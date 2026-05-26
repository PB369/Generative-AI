"""
🌿 DeforestWatch AI — Aplicação de Deploy
Monitoramento de Desmatamento e Queimadas via Satélite
Streamlit App
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib, json, os
from pathlib import Path
import shap

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeforestWatch AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; }
    .main-header p  { font-size: 1.05rem; opacity: 0.8; margin: 0.5rem 0 0; }

    .metric-card {
        background: #f0faf4;
        border: 1.5px solid #a8dbb4;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h3 { color: #1a6b3a; margin: 0; font-size: 1.7rem; }
    .metric-card p  { color: #555; margin: 0; font-size: 0.85rem; }

    .risk-alto   { background:#fff0f0; border:2px solid #e74c3c; border-radius:10px; padding:1rem; color:#c0392b; font-weight:700; font-size:1.4rem; text-align:center; }
    .risk-medio  { background:#fffbf0; border:2px solid #f39c12; border-radius:10px; padding:1rem; color:#e67e22; font-weight:700; font-size:1.4rem; text-align:center; }
    .risk-baixo  { background:#f0fff4; border:2px solid #27ae60; border-radius:10px; padding:1rem; color:#1e8449; font-weight:700; font-size:1.4rem; text-align:center; }

    .section-title { font-size:1.3rem; font-weight:700; color:#1a3c2a; border-left:4px solid #27ae60; padding-left:0.8rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Carregamento de artefatos ─────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    mdls = BASE / "models"
    rf   = joblib.load(mdls / "random_forest.pkl")
    xgb  = joblib.load(mdls / "xgboost.pkl")
    sc   = joblib.load(mdls / "scaler.pkl")
    rfe  = joblib.load(mdls / "rfe_selector.pkl")
    le   = joblib.load(mdls / "label_encoder.pkl")
    with open(mdls / "pipeline_meta.json") as f:
        meta = json.load(f)
    with open(mdls / "shap_importances.json") as f:
        shap_imp = json.load(f)
    return rf, xgb, sc, rfe, le, meta, shap_imp

rf, xgb, scaler, rfe, le, meta, shap_imp = load_artifacts()
SELECTED = meta["selected_features"]
ALL_FEATS = meta["all_features"]
CLASSES = meta["target_classes"]   # Alto, Baixo, Médio

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 DeforestWatch AI</h1>
    <p>Pipeline de Machine Learning para Monitoramento de Desmatamento e Queimadas via Satélite</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Navegação ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Deforestation_of_Amazon_rainforest.jpg/640px-Deforestation_of_Amazon_rainforest.jpg", use_container_width=True)
    st.markdown("---")
    page = st.radio("Navegação", [
        "🔮 Predição de Risco",
        "📊 Análise Exploratória",
        "🧠 Interpretabilidade SHAP",
        "🏆 Comparação de Modelos",
    ])
    st.markdown("---")
    st.markdown("**Melhor modelo:** XGBoost 🥇  \n**Acurácia:** 77.25%  \n**CV (5-fold):** 78.62% ± 2.23%")

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — PREDIÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predição de Risco":
    st.markdown('<div class="section-title">Insira os dados do ponto satélite</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        lat         = st.slider("Latitude",            -15.0, 5.0,   -5.0, 0.01)
        temp        = st.slider("Temperatura (°C)",     20.0, 45.0,  30.0, 0.1)
        umidade     = st.slider("Umidade Relativa (%)", 20.0, 100.0, 65.0, 0.5)
        ndvi        = st.slider("NDVI",                -0.2,  1.0,   0.5,  0.01)
    with col2:
        ndwi        = st.slider("NDWI",                -0.5,  0.8,   0.2,  0.01)
        precip      = st.slider("Precipitação 30d (mm)", 0.0, 300.0, 80.0, 1.0)
        frp         = st.slider("FRP — Fogo (MW)",      0.0,  50.0,  3.0,  0.5)
        dist_road   = st.slider("Dist. Estrada (km)",   0.0, 150.0, 25.0, 0.5)
    with col3:
        dist_prot   = st.slider("Dist. Área Protegida (km)", 0.0, 200.0, 40.0, 1.0)
        dias_sr     = st.slider("Dias sem Chuva",        0.0,  90.0, 15.0, 0.5)

    # Construir vetor de features na ordem de ALL_FEATS
    # (features não selecionadas → preencher com mediana típica)
    defaults = {
        "latitude": lat, "longitude": -60.0, "mes": 7,
        "temp_superficie": temp, "umidade_relativa": umidade,
        "ndvi": ndvi, "ndwi": ndwi, "precipitacao_30d": precip,
        "velocidade_vento": 12.0, "frp_fogo": frp,
        "dist_estrada_km": dist_road, "dist_area_protegida_km": dist_prot,
        "densidade_pop": 8.0, "altitude_m": 300.0,
        "dias_sem_chuva": dias_sr, "estacao_seca": int(precip < 50),
    }
    X_input = np.array([[defaults[f] for f in ALL_FEATS]])
    X_scaled = scaler.transform(X_input)
    X_sel    = X_scaled[:, rfe.support_]

    if st.button("🔍 Prever Risco de Desmatamento", type="primary", use_container_width=True):
        pred_xgb = xgb.predict(X_sel)[0]
        prob_xgb = xgb.predict_proba(X_sel)[0]
        pred_rf  = rf.predict(X_sel)[0]
        prob_rf  = rf.predict_proba(X_sel)[0]

        label_xgb = le.inverse_transform([pred_xgb])[0]
        label_rf  = le.inverse_transform([pred_rf])[0]

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🥇 XGBoost (Melhor Modelo)")
            css_class = f"risk-{label_xgb.lower()}"
            icon = "🔴" if label_xgb == "Alto" else ("🟡" if label_xgb == "Médio" else "🟢")
            st.markdown(f'<div class="{css_class}">{icon} Risco: {label_xgb.upper()}</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            colors = ["#e74c3c", "#27ae60", "#f39c12"]
            bars = ax.barh(CLASSES, prob_xgb * 100, color=colors, edgecolor="white", height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilidade (%)")
            ax.set_title("XGBoost — Probabilidades", fontsize=10, fontweight="bold")
            for b, v in zip(bars, prob_xgb * 100):
                ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("#### 🌲 Random Forest")
            label_rf_icon = "🔴" if label_rf == "Alto" else ("🟡" if label_rf == "Médio" else "🟢")
            css_rf = f"risk-{label_rf.lower()}"
            st.markdown(f'<div class="{css_rf}">{label_rf_icon} Risco: {label_rf.upper()}</div>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
            ax2.barh(CLASSES, prob_rf * 100, color=colors, edgecolor="white", height=0.5)
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Probabilidade (%)")
            ax2.set_title("Random Forest — Probabilidades", fontsize=10, fontweight="bold")
            for b, v in zip(ax2.patches, prob_rf * 100):
                ax2.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Análise Exploratória":
    st.markdown('<div class="section-title">Análise Exploratória dos Dados</div>', unsafe_allow_html=True)

    df = pd.read_csv(BASE / "data" / "dataset_desmatamento.csv")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total de Registros", f"{len(df):,}")
    col2.metric("📐 Features",          f"{df.shape[1]}")
    col3.metric("🔴 Risco Alto",        f"{(df['risco_desmatamento']=='Alto').sum()}")
    col4.metric("🟢 Risco Baixo",       f"{(df['risco_desmatamento']=='Baixo').sum()}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Distribuição das Classes**")
        counts = df["risco_desmatamento"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#e74c3c", "#27ae60", "#f39c12"]
        ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Quantidade")
        ax.set_title("Distribuição de Risco de Desmatamento")
        for i, (lbl, cnt) in enumerate(counts.items()):
            ax.text(i, cnt + 10, str(cnt), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**NDVI por Nível de Risco**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for risk, color in zip(["Alto", "Médio", "Baixo"], ["#e74c3c", "#f39c12", "#27ae60"]):
            vals = df[df["risco_desmatamento"] == risk]["ndvi"]
            ax.hist(vals, bins=25, alpha=0.65, label=risk, color=color, edgecolor="white")
        ax.set_xlabel("NDVI")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição de NDVI por Risco")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Correlação de Pearson — Features × Target**")
    corr_img = BASE / "docs" / "figures" / "pearson_correlation.png"
    if corr_img.exists():
        st.image(str(corr_img), use_container_width=True)

    st.markdown("**Amostra do Dataset**")
    st.dataframe(df.head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Interpretabilidade SHAP":
    st.markdown('<div class="section-title">Interpretabilidade com SHAP</div>', unsafe_allow_html=True)
    st.info("SHAP (SHapley Additive exPlanations) explica a contribuição de cada variável na decisão do modelo XGBoost.")

    # SHAP importâncias (bar)
    st.markdown("**Importância Média Absoluta (SHAP)**")
    shap_bar_img = BASE / "docs" / "figures" / "shap_bar.png"
    shap_sum_img = BASE / "docs" / "figures" / "shap_summary.png"

    c1, c2 = st.columns(2)
    with c1:
        if shap_bar_img.exists():
            st.image(str(shap_bar_img), use_container_width=True)
    with c2:
        if shap_sum_img.exists():
            st.image(str(shap_sum_img), use_container_width=True)

    st.markdown("---")
    st.markdown("**Ranking de Importância SHAP**")
    df_shap = pd.DataFrame({"Feature": list(shap_imp.keys()), "Importância SHAP": list(shap_imp.values())})
    df_shap = df_shap.sort_values("Importância SHAP", ascending=False).reset_index(drop=True)
    df_shap.index += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(df_shap)))
    ax.barh(df_shap["Feature"][::-1], df_shap["Importância SHAP"][::-1], color=colors[::-1])
    ax.set_xlabel("Importância SHAP média (|valor|)")
    ax.set_title("Ranking de Importância das Features — XGBoost", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(df_shap, use_container_width=True)

    st.markdown("""
    #### 💡 Interpretação
    - **temp_superficie** é a variável mais influente — temperaturas elevadas (épocas secas) aumentam drasticamente o risco.
    - **ndwi** (índice hídrico) reflete a disponibilidade de água na vegetação.
    - **umidade_relativa** e **ndvi** confirmam que áreas secas e desmatadas têm maior risco.
    - **frp_fogo** (Fire Radiative Power) detecta focos de calor ativos via satélite.
    - **dist_area_protegida_km** — áreas distantes de unidades de conservação têm maior risco.
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Comparação de Modelos":
    st.markdown('<div class="section-title">Comparação dos Modelos</div>', unsafe_allow_html=True)

    results = meta.get("results", [])
    df_r = pd.DataFrame(results)

    # Métricas lado a lado
    cols = st.columns(len(results))
    for i, (col, row) in enumerate(zip(cols, results)):
        with col:
            is_best = row["model"] == meta["best_model"]
            st.markdown(f"### {'🥇 ' if is_best else '🌲 '}{row['model']}")
            st.metric("Acurácia Teste",        f"{row['accuracy_test']*100:.2f}%")
            st.metric("CV Média (5-fold)",      f"{row['cv_mean']*100:.2f}%")
            st.metric("CV Desvio Padrão",       f"±{row['cv_std']*100:.2f}%")
            if is_best:
                st.success("✅ Modelo selecionado para deploy")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Matriz de Confusão — Random Forest**")
        rf_cm = BASE / "docs" / "figures" / "confusion_matrix_random_forest.png"
        if rf_cm.exists():
            st.image(str(rf_cm), use_container_width=True)

    with c2:
        st.markdown("**Matriz de Confusão — XGBoost**")
        xgb_cm = BASE / "docs" / "figures" / "confusion_matrix_xgboost.png"
        if xgb_cm.exists():
            st.image(str(xgb_cm), use_container_width=True)

    st.markdown("---")
    st.markdown("**Features selecionadas por RFE**")
    st.code(", ".join(meta["selected_features"]))

    st.markdown("""
    #### 📝 Conclusão
    O **XGBoost** superou o Random Forest em acurácia de teste (77.25% vs 76.00%).
    Ambos os modelos foram validados com StratifiedKFold (5-fold), garantindo robustez.
    O XGBoost foi selecionado para deploy por apresentar melhor acurácia e F1-score balanceado entre as classes.
    """)
