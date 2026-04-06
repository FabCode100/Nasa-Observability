"""
══════════════════════════════════════════════════════════════
TCC Astrofísica — Dashboard Interativo
Detecção de Anomalias Estelares em Dados do Telescópio Kepler

Execução:
    streamlit run dashboard.py
══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Configuração da página ───────────────────────────────
st.set_page_config(
    page_title="Anomalias Estelares — Kepler/NASA",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado (tema escuro astronômico) ──────────
st.markdown(f"""
<style>
    .stApp {{
        background-color: {config.THEME['bg']};
    }}
    .main .block-container {{
        padding-top: 1rem;
        max-width: 1400px;
    }}
    h1, h2, h3, h4, h5, h6, p, li, span, label, div {{
        color: {config.THEME['text']} !important;
    }}
    .metric-card {{
        background-color: {config.THEME['panel_bg']};
        border: 1px solid {config.THEME['grid']};
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
    }}
    .metric-label {{
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.3rem;
    }}
    .anomaly {{
        color: {config.THEME['accent3']} !important;
    }}
    .normal {{
        color: {config.THEME['accent2']} !important;
    }}
    .section-title {{
        border-left: 4px solid {config.THEME['accent1']};
        padding-left: 12px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    .stSidebar [data-testid="stSidebarContent"] {{
        background-color: {config.THEME['panel_bg']};
    }}
</style>
""", unsafe_allow_html=True)


def load_results():
    """Carrega os resultados do pipeline."""
    results_path = os.path.join(config.RESULTS_DIR, "pipeline_results.json")
    if not os.path.exists(results_path):
        st.error(
            "⚠️ Resultados não encontrados. Execute primeiro:\n\n"
            "```\npython main.py\n```"
        )
        st.stop()
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plotly_layout(title="", height=500):
    """Layout padrão para gráficos Plotly com tema dark."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=config.THEME["text"])),
        paper_bgcolor=config.THEME["bg"],
        plot_bgcolor=config.THEME["panel_bg"],
        font=dict(color=config.THEME["text"], size=12),
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(
            gridcolor=config.THEME["grid"],
            zerolinecolor=config.THEME["grid"],
        ),
        yaxis=dict(
            gridcolor=config.THEME["grid"],
            zerolinecolor=config.THEME["grid"],
        ),
        legend=dict(
            bgcolor=config.THEME["panel_bg"],
            bordercolor=config.THEME["grid"],
            borderwidth=1,
            font=dict(size=11),
        ),
    )


def render_metric_card(label, value, css_class=""):
    """Renderiza um card de métrica."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {css_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════
data = load_results()

target_star = data["target_star"]
target_proc = data["target_processed"]
ref_stars = data.get("reference_processed", [])
summary = data["model_summary"]
target_result = summary["target"]

pca_coords = np.array(data["pca_coords"])
if_scores = np.array(data["if_scores"])
if_preds = np.array(data["if_predictions"])
cluster_labels = np.array(data["cluster_labels"])
feature_matrix = np.array(data["feature_matrix"])
pca_var = data["pca_variance"]
names = data["names"]
labels = data["labels"]
feature_names = data["feature_names"]

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔭 Configurações")
    st.markdown("---")

    st.markdown(f"**Estrela Alvo:** {config.TARGET_STAR['kic_id']}")
    st.markdown(f"**Nome:** {config.TARGET_STAR['nome']}")
    st.markdown(f"**Quarter:** {config.TARGET_STAR['quarter']}")

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown(f"- Total: **{summary['n_total']}** estrelas")
    st.markdown(f"- Anomalias: **{summary['n_anomalies']}**")
    st.markdown(f"- Reais: **{len(ref_stars) + 1}**")
    st.markdown(f"- Simuladas: **{summary['n_total'] - len(ref_stars) - 1}**")

    st.markdown("---")
    st.markdown("### 🧪 Modelos")
    st.markdown(f"- Isolation Forest: **{config.ISOLATION_FOREST['n_estimators']}** árvores")
    st.markdown(f"- K-Means: **k={config.KMEANS['n_clusters']}**")
    st.markdown(f"- Silhouette: **{summary['silhouette_score']:.3f}**")

    st.markdown("---")
    show_ref_curves = st.checkbox("Mostrar curvas de referência", value=True)
    show_simulated = st.checkbox("Mostrar pontos simulados", value=True)

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
# 🌟 Detecção de Anomalias Estelares
### Telescópio Kepler / NASA — Isolation Forest + K-Means
""")

# ── Cards de Métricas ────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    status = "ANOMALIA ⚠️" if target_result["is_anomaly"] else "NORMAL ✓"
    css = "anomaly" if target_result["is_anomaly"] else "normal"
    render_metric_card("Isolation Forest", status, css)

with col2:
    render_metric_card("Score de Anomalia", f"{target_result['if_score']:.4f}")

with col3:
    tc = target_result["cluster"]
    hyp = config.CLUSTER_HYPOTHESES.get(tc, {})
    render_metric_card("Cluster K-Means", f"C{tc}: {hyp.get('nome', '?')}")

with col4:
    render_metric_card("Silhouette Score", f"{summary['silhouette_score']:.3f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAINEL 1: Curva de Luz Interativa
# ══════════════════════════════════════════════════════════
st.markdown('<h2 class="section-title">📈 Curva de Luz — Estrela de Tabby</h2>',
            unsafe_allow_html=True)

fig_lc = go.Figure()

# Curva do alvo
fig_lc.add_trace(go.Scattergl(
    x=target_proc["time"],
    y=target_proc["flux"],
    mode="markers",
    marker=dict(size=2, color=config.THEME["accent1"], opacity=0.6),
    name=f"{target_proc['kic_id']} (Alvo)",
    hovertemplate="Tempo: %{x:.2f} d<br>Fluxo: %{y:.6f}<extra></extra>",
))

# Linha de referência
fig_lc.add_hline(
    y=1.0,
    line_dash="dash",
    line_color=config.THEME["accent2"],
    annotation_text="Fluxo de referência (1.0)",
    annotation_font_color=config.THEME["accent2"],
)

# Curvas de referência
if show_ref_curves and ref_stars:
    ref_colors = {"estaveis": "#4488cc", "variaveis": "#44aa66", "anomalas": "#cc6666"}
    for star in ref_stars:
        fig_lc.add_trace(go.Scattergl(
            x=star["time"],
            y=star["flux"],
            mode="markers",
            marker=dict(size=1.5, color=ref_colors.get(star.get("categoria"), "#888"), opacity=0.3),
            name=f"{star['kic_id']} ({star.get('categoria', '')})",
            visible="legendonly",
            hovertemplate=f"{star['kic_id']}<br>Tempo: %{{x:.2f}} d<br>Fluxo: %{{y:.6f}}<extra></extra>",
        ))

fig_lc.update_layout(**plotly_layout("Curva de Luz — Fluxo Normalizado vs. Tempo", height=450))
fig_lc.update_xaxes(title_text="Tempo (dias BJD)")
fig_lc.update_yaxes(title_text="Fluxo Normalizado")
st.plotly_chart(fig_lc, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAINEL 2 e 3: PCA — Isolation Forest + K-Means
# ══════════════════════════════════════════════════════════
st.markdown('<h2 class="section-title">🔍 Espaço PCA — Análise de Anomalias e Clusters</h2>',
            unsafe_allow_html=True)

col_pca1, col_pca2 = st.columns(2)

with col_pca1:
    # PCA + Isolation Forest
    fig_if = go.Figure()

    # Normais
    norm_mask = if_preds[:-1] == 1
    fig_if.add_trace(go.Scatter(
        x=pca_coords[:-1][norm_mask, 0],
        y=pca_coords[:-1][norm_mask, 1],
        mode="markers",
        marker=dict(size=5, color=config.THEME["accent1"], opacity=0.3),
        name="Normal",
        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra>Normal</extra>",
    ))

    # Anomalias
    anom_mask = if_preds[:-1] == -1
    anom_names = [names[i] for i in range(len(names)-1) if anom_mask[i]]
    fig_if.add_trace(go.Scatter(
        x=pca_coords[:-1][anom_mask, 0],
        y=pca_coords[:-1][anom_mask, 1],
        mode="markers",
        marker=dict(size=8, color=config.THEME["accent3"], opacity=0.7),
        name="Anomalia",
        text=anom_names,
        hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra>Anomalia</extra>",
    ))

    # Alvo
    fig_if.add_trace(go.Scatter(
        x=[pca_coords[-1, 0]],
        y=[pca_coords[-1, 1]],
        mode="markers",
        marker=dict(
            size=18, color=config.THEME["target"],
            symbol="star", line=dict(width=1, color="white"),
        ),
        name=config.TARGET_STAR["kic_id"],
        hovertemplate=f"{config.TARGET_STAR['kic_id']}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra>ALVO</extra>",
    ))

    fig_if.update_layout(**plotly_layout(
        f"Isolation Forest (PCA: {pca_var[0]:.0%} + {pca_var[1]:.0%})", height=450
    ))
    fig_if.update_xaxes(title_text=f"PC1 ({pca_var[0]:.1%})")
    fig_if.update_yaxes(title_text=f"PC2 ({pca_var[1]:.1%})")
    st.plotly_chart(fig_if, use_container_width=True)

with col_pca2:
    # PCA + K-Means
    fig_km = go.Figure()

    colors = config.THEME["cluster_colors"]
    cnames = config.THEME["cluster_names"]

    for c in range(config.KMEANS["n_clusters"]):
        mask = cluster_labels[:-1] == c
        c_names_list = [names[i] for i in range(len(names)-1) if mask[i]]
        fig_km.add_trace(go.Scatter(
            x=pca_coords[:-1][mask, 0],
            y=pca_coords[:-1][mask, 1],
            mode="markers",
            marker=dict(size=5, color=colors[c], opacity=0.4),
            name=cnames[c],
            text=c_names_list,
            hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra>" + cnames[c] + "</extra>",
        ))

    fig_km.add_trace(go.Scatter(
        x=[pca_coords[-1, 0]],
        y=[pca_coords[-1, 1]],
        mode="markers",
        marker=dict(
            size=18, color=config.THEME["target"],
            symbol="star", line=dict(width=1, color="white"),
        ),
        name=f"{config.TARGET_STAR['kic_id']} (C{target_result['cluster']})",
    ))

    fig_km.update_layout(**plotly_layout("K-Means (k=3)", height=450))
    fig_km.update_xaxes(title_text=f"PC1 ({pca_var[0]:.1%})")
    fig_km.update_yaxes(title_text=f"PC2 ({pca_var[1]:.1%})")
    st.plotly_chart(fig_km, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAINEL 4: Scores de Anomalia e Feature Radar
# ══════════════════════════════════════════════════════════
st.markdown('<h2 class="section-title">📊 Scores de Anomalia e Perfil de Features</h2>',
            unsafe_allow_html=True)

col_sc, col_radar = st.columns(2)

with col_sc:
    # Histograma de scores + posição do alvo
    fig_hist = go.Figure()

    fig_hist.add_trace(go.Histogram(
        x=if_scores[:-1],
        nbinsx=50,
        marker_color=config.THEME["accent1"],
        opacity=0.6,
        name="População",
    ))

    # Linha vertical para o alvo
    fig_hist.add_vline(
        x=target_result["if_score"],
        line_dash="dash",
        line_color=config.THEME["target"],
        annotation_text=f"Alvo: {target_result['if_score']:.4f}",
        annotation_font_color=config.THEME["target"],
    )

    # Limiar 5%
    threshold = summary["anomaly_threshold"]
    fig_hist.add_vline(
        x=threshold,
        line_dash="dot",
        line_color=config.THEME["accent3"],
        annotation_text=f"Limiar 5%: {threshold:.4f}",
        annotation_font_color=config.THEME["accent3"],
        annotation_position="top left",
    )

    fig_hist.update_layout(**plotly_layout("Distribuição dos Scores de Anomalia", height=420))
    fig_hist.update_xaxes(title_text="Score (mais negativo = mais anômalo)")
    fig_hist.update_yaxes(title_text="Contagem")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_radar:
    # Radar Chart (perfil de features do alvo vs. média dos clusters)
    target_feats = target_star.get("features", {})
    target_vals = [target_feats.get(fn, 0) for fn in feature_names]

    # Normalizar para [0, 1] para o radar
    feat_matrix = np.array(data["feature_matrix"])
    feat_min = feat_matrix.min(axis=0)
    feat_max = feat_matrix.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1

    target_norm = [(v - mn) / rng for v, mn, rng in zip(target_vals, feat_min, feat_range)]

    fig_radar = go.Figure()

    # Média por cluster
    for c in range(config.KMEANS["n_clusters"]):
        mask = cluster_labels == c
        cluster_mean = feat_matrix[mask].mean(axis=0)
        cluster_norm = [(v - mn) / rng for v, mn, rng in zip(cluster_mean, feat_min, feat_range)]
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_norm + [cluster_norm[0]],
            theta=feature_names + [feature_names[0]],
            fill="toself",
            name=cnames[c],
            line=dict(color=colors[c]),
            opacity=0.3,
        ))

    # Alvo
    fig_radar.add_trace(go.Scatterpolar(
        r=target_norm + [target_norm[0]],
        theta=feature_names + [feature_names[0]],
        fill="toself",
        name=config.TARGET_STAR["kic_id"],
        line=dict(color=config.THEME["target"], width=3),
        opacity=0.7,
    ))

    fig_radar.update_layout(
        **plotly_layout("Perfil de Features (Radar)", height=420),
        polar=dict(
            bgcolor=config.THEME["panel_bg"],
            angularaxis=dict(gridcolor=config.THEME["grid"], color=config.THEME["text"]),
            radialaxis=dict(
                gridcolor=config.THEME["grid"],
                color=config.THEME["text"],
                range=[0, 1.1],
            ),
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAINEL 5: Espaço de Features 3D
# ══════════════════════════════════════════════════════════
st.markdown('<h2 class="section-title">🌌 Espaço de Features 3D (STD × Range × Skewness)</h2>',
            unsafe_allow_html=True)

fig_3d = go.Figure()

for c in range(config.KMEANS["n_clusters"]):
    mask = cluster_labels[:-1] == c
    fig_3d.add_trace(go.Scatter3d(
        x=feature_matrix[:-1][mask, 0],  # std
        y=feature_matrix[:-1][mask, 1],  # range
        z=feature_matrix[:-1][mask, 2],  # skewness
        mode="markers",
        marker=dict(size=3, color=colors[c], opacity=0.4),
        name=cnames[c],
    ))

# Alvo
fig_3d.add_trace(go.Scatter3d(
    x=[feature_matrix[-1, 0]],
    y=[feature_matrix[-1, 1]],
    z=[feature_matrix[-1, 2]],
    mode="markers",
    marker=dict(size=12, color=config.THEME["target"], symbol="diamond"),
    name=config.TARGET_STAR["kic_id"],
))

fig_3d.update_layout(
    **plotly_layout("Features: STD × Range × Skewness", height=550),
    scene=dict(
        xaxis=dict(title="STD", backgroundcolor=config.THEME["panel_bg"],
                   gridcolor=config.THEME["grid"], color=config.THEME["text"]),
        yaxis=dict(title="Range", backgroundcolor=config.THEME["panel_bg"],
                   gridcolor=config.THEME["grid"], color=config.THEME["text"]),
        zaxis=dict(title="Skewness", backgroundcolor=config.THEME["panel_bg"],
                   gridcolor=config.THEME["grid"], color=config.THEME["text"]),
        bgcolor=config.THEME["panel_bg"],
    ),
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAINEL 6: Relatório de Anomalia
# ══════════════════════════════════════════════════════════
st.markdown('<h2 class="section-title">📋 Relatório de Anomalia</h2>',
            unsafe_allow_html=True)

col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    st.markdown(f"""
    ### {config.TARGET_STAR['kic_id']} — {config.TARGET_STAR['nome']}

    **Descrição:** {config.TARGET_STAR['descricao']}

    ---

    | Feature | Valor |
    |---------|-------|
    | Desvio Padrão (std) | `{target_feats.get('std', 0):.6f}` |
    | Amplitude (range) | `{target_feats.get('range', 0):.6f}` |
    | Assimetria (skewness) | `{target_feats.get('skewness', 0):.6f}` |
    | MAD | `{target_feats.get('mad', 0):.6f}` |
    | Curtose (kurtosis) | `{target_feats.get('kurtosis', 0):.6f}` |
    | Fração < P2 | `{target_feats.get('below_p2', 0):.6f}` |
    """)

with col_rep2:
    st.markdown(f"""
    ### Resultado dos Modelos

    - **Isolation Forest:** {'🔴 ANOMALIA' if target_result['is_anomaly'] else '🟢 NORMAL'}
      (score = `{target_result['if_score']:.4f}`)
    - **Cluster K-Means:** C{tc} — {hyp.get('nome', '?')}
    - **Silhouette Score:** `{summary['silhouette_score']:.3f}`
    - **Variância PCA:** `{pca_var[0]:.1%}` + `{pca_var[1]:.1%}` = `{sum(pca_var):.1%}`

    ---

    ### Hipótese Principal

    > {hyp.get('hipotese', 'N/A')}

    ---

    ### Descrição do Cluster

    > {hyp.get('descricao', 'N/A')}
    """)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<center style='opacity:0.5'>TCC Astrofísica — Detecção de Anomalias Estelares — "
    "Kepler/NASA — Isolation Forest + K-Means</center>",
    unsafe_allow_html=True,
)
