# 🌿 DeforestWatch AI — Monitoramento de Desmatamento e Queimadas via Satélite

> Pipeline completo de Machine Learning para classificação de risco de desmatamento e queimadas com dados sintéticos de sensoriamento remoto.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Deploy-Streamlit-red)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/Interpretability-SHAP-green)](https://shap.readthedocs.io)

---

## 📌 Contexto do Problema

O desmatamento e as queimadas na Amazônia e no Cerrado representam uma das maiores crises ambientais da atualidade, com impactos diretos na biodiversidade, no ciclo hidrológico e nas emissões de carbono. Satélites como o Landsat, Sentinel-2 e MODIS fornecem dados multiespectrais em tempo quase real, possibilitando o monitoramento contínuo da cobertura vegetal.

Este projeto simula um sistema de **alerta precoce** que classifica pontos geográficos em três níveis de risco — **Baixo, Médio e Alto** — com base em variáveis derivadas de imagens de satélite e dados climáticos.

---

## 📦 Fonte dos Dados

Dataset **sintético** gerado com IA generativa (`data/generate_dataset.py`):

| Variável | Descrição |
|---|---|
| `latitude / longitude` | Coordenadas no bioma Amazônia/Cerrado |
| `mes` | Mês da observação |
| `temp_superficie` | Temperatura superficial em °C (LST) |
| `umidade_relativa` | Umidade relativa do ar (%) |
| `ndvi` | Normalized Difference Vegetation Index |
| `ndwi` | Normalized Difference Water Index |
| `precipitacao_30d` | Precipitação acumulada em 30 dias (mm) |
| `velocidade_vento` | Velocidade do vento (km/h) |
| `frp_fogo` | Fire Radiative Power — intensidade de fogo (MW) |
| `dist_estrada_km` | Distância à estrada mais próxima (km) |
| `dist_area_protegida_km` | Distância à unidade de conservação (km) |
| `densidade_pop` | Densidade populacional (hab/km²) |
| `altitude_m` | Altitude (m) |
| `dias_sem_chuva` | Dias consecutivos sem chuva |
| `estacao_seca` | Flag de estação seca (jun-set) |
| `risco_desmatamento` | **TARGET** — Baixo / Médio / Alto |

- **2.000 registros × 19 colunas**
- Distribuição: Alto 14% · Médio 44% · Baixo 42%

---

## ⚙️ Metodologia

```
Geração dos Dados
        ↓
Pré-processamento (LabelEncoder, StandardScaler, dropna)
        ↓
Análise de Pearson (correlação feature × target)
        ↓
Seleção de Features — RFE com RandomForest (top-10)
        ↓
Train/Test Split 80/20 estratificado
        ↓
Treinamento: Random Forest + XGBoost
        ↓
Validação: Acurácia · Matriz de Confusão · CV 5-fold
        ↓
Comparação e escolha do melhor modelo
        ↓
Interpretabilidade: SHAP TreeExplainer
        ↓
Deploy: Streamlit App
```

---

## 🤖 Modelos Testados

### Random Forest
- `n_estimators=300`, `max_depth=12`, `class_weight='balanced'`
- Acurácia Teste: **76.00%**
- CV 5-fold: **79.81% ± 1.83%**

### XGBoost ✅ Melhor Modelo
- `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`
- Acurácia Teste: **77.25%**
- CV 5-fold: **78.62% ± 2.23%**

| Modelo | Acurácia Teste | CV Média | CV Std |
|---|---|---|---|
| Random Forest | 76.00% | 79.81% | ±1.83% |
| **XGBoost** | **77.25%** | **78.62%** | ±2.23% |

---

## 📊 Resultados

### XGBoost — Classification Report

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Alto | 0.68 | 0.77 | 0.72 |
| Baixo | 0.84 | 0.80 | 0.82 |
| Médio | 0.74 | 0.75 | 0.74 |
| **Accuracy** | | | **0.77** |

---

## 🧠 Interpretação com SHAP

Utilizou-se o `shap.TreeExplainer` aplicado ao XGBoost. O gráfico de importância média absoluta revelou:

| Rank | Feature | SHAP (mean |value|) | Interpretação |
|---|---|---|---|
| 1 | `temp_superficie` | 1.63 | LST elevada = vegetação estressada → alto risco |
| 2 | `ndwi` | 0.48 | Baixo NDWI = baixa umidade da vegetação |
| 3 | `umidade_relativa` | 0.35 | Baixa umidade do ar potencializa queimadas |
| 4 | `ndvi` | 0.32 | Menor NDVI = menor cobertura vegetal |
| 5 | `dist_area_protegida_km` | 0.27 | Longe de UCs → pressão antrópica maior |
| 6 | `frp_fogo` | 0.25 | FRP alto = foco de calor ativo detectado |
| 7 | `dist_estrada_km` | 0.18 | Perto de estradas = maior atividade humana |
| 8 | `precipitacao_30d` | 0.13 | Baixa chuva acumulada favorece ignição |
| 9 | `dias_sem_chuva` | 0.10 | Mais dias sem chuva → material combustível seco |
| 10 | `latitude` | 0.10 | Variação geográfica no bioma |

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/deforestation-ai.git
cd deforestation-ai
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Gere o dataset

```bash
python data/generate_dataset.py
```

### 4. Execute o pipeline completo

```bash
python pipeline.py
```

Este comando irá:
- Pré-processar os dados
- Calcular correlações de Pearson
- Executar RFE para seleção de features
- Treinar Random Forest e XGBoost
- Gerar matrizes de confusão
- Gerar gráficos SHAP
- Salvar todos os artefatos em `/models/`

### 5. Iniciar a aplicação Streamlit

```bash
streamlit run app.py
```

Acesse em: `http://localhost:8501`

---

## 📁 Estrutura do Projeto

```
deforestation-ai/
├── data/
│   ├── generate_dataset.py       # Geração do dataset sintético
│   └── dataset_desmatamento.csv  # Dataset gerado (2000×19)
├── models/
│   ├── random_forest.pkl         # Modelo Random Forest treinado
│   ├── xgboost.pkl               # Modelo XGBoost treinado
│   ├── scaler.pkl                # StandardScaler
│   ├── rfe_selector.pkl          # RFE selector
│   ├── label_encoder.pkl         # LabelEncoder
│   ├── pipeline_meta.json        # Metadados do pipeline
│   └── shap_importances.json     # Importâncias SHAP
├── docs/
│   └── figures/
│       ├── pearson_correlation.png
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_xgboost.png
│       ├── shap_summary.png
│       └── shap_bar.png
├── pipeline.py                   # Pipeline completo de ML
├── app.py                        # Aplicação Streamlit
├── requirements.txt
└── README.md
```

---

## 🌐 Link da Aplicação

> Deploy disponível em: **https://deforestation-ai.streamlit.app** *(após publicar no Streamlit Cloud)*

Para deploy no Streamlit Cloud:
1. Faça push do repositório para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte o repositório e aponte para `app.py`

---

## 📚 Referências

- PRODES / INPE — Sistema de monitoramento do desmatamento
- NASA FIRMS — Fire Information for Resource Management System
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost. *KDD '16*.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

*Projeto desenvolvido para a disciplina Generative AI For Engineering (GAIE)*
