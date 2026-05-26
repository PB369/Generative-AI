"""
Pipeline Completo de ML — Monitoramento de Desmatamento e Queimadas
Técnicas: Random Forest + XGBoost
Métricas: Acurácia, Matriz de Confusão, Pearson, RFE, SHAP
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA  = BASE / "data"
MDLS  = BASE / "models"
FIGS  = BASE / "docs" / "figures"
MDLS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA E PRÉ-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CARGA E PRÉ-PROCESSAMENTO")
print("=" * 60)

df = pd.read_csv(DATA / "dataset_desmatamento.csv")
print(f"Shape original: {df.shape}")
print(df.dtypes)

# Remover colunas não usadas como features
df = df.drop(columns=["data", "risco_score"])

# Encode target
le = LabelEncoder()
df["target"] = le.fit_transform(df["risco_desmatamento"])  # Alto=0, Baixo=1, Médio=2
df = df.drop(columns=["risco_desmatamento"])

print(f"\nClasses: {list(le.classes_)}")
print(f"Distribuição:\n{df['target'].value_counts()}")

# Verificar nulos
print(f"\nNulos: {df.isnull().sum().sum()}")
df = df.dropna()

# ── Features e target ──────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != "target"]
X = df[FEATURE_COLS]
y = df["target"]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENGENHARIA E SELEÇÃO DE ATRIBUTOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. ENGENHARIA E SELEÇÃO DE ATRIBUTOS")
print("=" * 60)

# Pearson Correlation com target
pearson_corr = X.corrwith(y.astype(float)).sort_values(key=abs, ascending=False)
print("\nCorrelação de Pearson (top 10):")
print(pearson_corr.head(10).to_string())

# Salvar gráfico de correlação
fig, ax = plt.subplots(figsize=(10, 6))
pearson_corr.plot(kind="bar", ax=ax, color=pearson_corr.apply(lambda v: "#e74c3c" if v > 0 else "#2980b9"))
ax.set_title("Correlação de Pearson — Features × Target", fontsize=13, fontweight="bold")
ax.set_ylabel("Correlação")
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(FIGS / "pearson_correlation.png", dpi=150)
plt.close()
print("Gráfico salvo: pearson_correlation.png")

# RFE com Random Forest base
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf_base, n_features_to_select=10, step=1)
rfe.fit(X_scaled, y)
selected_features = [f for f, s in zip(FEATURE_COLS, rfe.support_) if s]
print(f"\nFeatures selecionadas por RFE ({len(selected_features)}): {selected_features}")

X_sel = X[selected_features]
X_sel_scaled = X_scaled[:, rfe.support_]

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_sel_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreino: {X_train.shape} | Teste: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TREINAMENTO DOS MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. TREINAMENTO DOS MODELOS")
print("=" * 60)

# ── Random Forest ──────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("Random Forest treinado.")

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1, verbosity=0
)
xgb.fit(X_train, y_train)
print("XGBoost treinado.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. VALIDAÇÃO E COMPARAÇÃO DE DESEMPENHO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. VALIDAÇÃO E COMPARAÇÃO DE DESEMPENHO")
print("=" * 60)

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"\n{'─'*40}")
    print(f"Modelo: {name}")
    print(f"Acurácia Teste:       {acc:.4f}")
    print(f"CV Acurácia (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nRelatório de Classificação:\n{classification_report(y_te, y_pred, target_names=le.classes_)}")

    # Matriz de confusão
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusão — {name}", fontweight="bold")
    plt.tight_layout()
    fname = FIGS / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Matriz de confusão salva: {fname.name}")

    return {"model": name, "accuracy_test": round(acc, 4), "cv_mean": round(cv_scores.mean(), 4), "cv_std": round(cv_scores.std(), 4)}

results = []
results.append(evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test))
results.append(evaluate_model("XGBoost", xgb, X_train, X_test, y_train, y_test))

# ── Comparação ────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
print("\n" + "=" * 60)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 60)
print(df_results.to_string(index=False))

best = df_results.loc[df_results["accuracy_test"].idxmax(), "model"]
print(f"\nMelhor modelo: {best}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTERPRETABILIDADE COM SHAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. INTERPRETABILIDADE COM SHAP")
print("=" * 60)

best_model = xgb if best == "XGBoost" else rf

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Normalizar: pode ser ndarray 3D (n,feat,classes) ou lista ou 2D
sv_arr = np.array(shap_values)
if sv_arr.ndim == 3:
    # shape (n_samples, n_features, n_classes) → média sobre amostras e classes
    shap_vals_summary = sv_arr[:, :, 1]          # classe "Médio" para plot detalhado
    mean_abs_shap = np.abs(sv_arr).mean(axis=(0, 2))  # média global
elif isinstance(shap_values, list):
    shap_vals_summary = shap_values[1]
    mean_abs_shap = np.array([np.abs(s).mean(axis=0) for s in shap_values]).mean(axis=0)
else:
    shap_vals_summary = sv_arr
    mean_abs_shap = np.abs(sv_arr).mean(axis=0)

# Summary plot
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_vals_summary, X_test, feature_names=selected_features, show=False)
plt.title(f"SHAP — Importância das Features ({best})", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(FIGS / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP summary plot salvo.")

# Bar plot
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_vals_summary, X_test, feature_names=selected_features, plot_type="bar", show=False)
plt.title(f"SHAP — Importância Média Absoluta ({best})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP bar plot salvo.")

mean_shap = pd.Series(mean_abs_shap, index=selected_features).sort_values(ascending=False)
shap_dict = mean_shap.round(4).to_dict()
with open(MDLS / "shap_importances.json", "w") as f:
    json.dump(shap_dict, f, indent=2)
print(f"SHAP importâncias: {shap_dict}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SALVAR ARTEFATOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SALVANDO ARTEFATOS")
print("=" * 60)

joblib.dump(rf,  MDLS / "random_forest.pkl")
joblib.dump(xgb, MDLS / "xgboost.pkl")
joblib.dump(scaler, MDLS / "scaler.pkl")
joblib.dump(rfe,    MDLS / "rfe_selector.pkl")
joblib.dump(le,     MDLS / "label_encoder.pkl")

meta = {
    "selected_features": selected_features,
    "all_features": FEATURE_COLS,
    "target_classes": list(le.classes_),
    "best_model": best,
    "results": results,
}
with open(MDLS / "pipeline_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Artefatos salvos em /models/")
print("\nPipeline concluído com sucesso!")
