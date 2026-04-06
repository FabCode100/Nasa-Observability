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
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Configuração da página ───────────────────────────────
st.set_page_config(
    page_title="Observabilidade — TCC AstroFísica & Indústria",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Definição do Tema Ativo ──────────────────────────────
# Usando o config.DOMAIN_ACTIVE como padrão inicial
active_domain = config.DOMAIN_ACTIVE
theme = config.THEME if active_domain == "SPACE" else config.THEME_INDUSTRIAL

active_bg = theme['bg']
active_panel = theme['panel_bg']
active_text = theme['text']
active_grid = theme['grid']
active_accent = theme['accent1']
active_accent2 = theme['accent2']
active_accent3 = theme['accent3']
active_target = theme['target']

# ── CSS personalizado (tema escuro astronômico) ──────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Outfit:wght@300;600;800&display=swap');
    
    .stApp {{
        background-color: {active_bg};
        font-family: 'Inter', sans-serif;
    }}
    .main .block-container {{
        padding-top: 2rem;
        max-width: 1500px;
    }}
    h1 {{
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }}
    h2 {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
    }}
    h3, h4, h5, h6, p, li, span, label, div, table, tr, td, th, blockquote {{
        color: {active_text} !important;
        font-size: 1.1rem;
    }}
    .report-table {{
        font-size: 1.6rem !important;
        width: 100%;
        margin-top: 1.5rem;
    }}
    .report-table td, .report-table th {{
        padding: 15px 20px !important;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .report-text {{
        font-size: 1.6rem !important;
        line-height: 1.8 !important;
    }}
    blockquote {{
        font-size: 1.8rem !important;
        border-left: 8px solid {active_accent} !important;
        padding: 1.5rem 2rem !important;
        background: rgba(255,255,255,0.05);
        border-radius: 0 15px 15px 0;
        font-style: italic;
    }}
    .metric-card {{
        background-color: {active_panel};
        border: 1px solid {active_grid};
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: {active_accent};
    }}
    .metric-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .metric-label {{
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
        margin-top: 0.5rem;
    }}
    .anomaly {{
        color: {active_accent3} !important;
        text-shadow: 0 0 15px rgba(255, 123, 114, 0.4);
    }}
    .normal {{
        color: {active_accent2} !important;
        text-shadow: 0 0 15px rgba(63, 185, 80, 0.4);
    }}
    .section-title {{
        border-left: 6px solid {active_accent};
        padding-left: 18px;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        font-family: 'Outfit', sans-serif;
    }}
    .stSidebar [data-testid="stSidebarContent"] {{
        background-color: {active_panel};
    }}
</style>
""", unsafe_allow_html=True)


def load_results(domain="SPACE"):
    """Carrega os resultados do pipeline baseado no domínio."""
    if domain == "SPACE":
        results_path = os.path.join(config.RESULTS_DIR, "pipeline_results.json")
        if not os.path.exists(results_path):
            st.error("⚠️ Resultados Space não encontrados. Execute `python main.py` (DOMAIN_ACTIVE='SPACE')")
            st.stop()
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        results_path = os.path.join(config.RESULTS_DIR, "industrial_results.pkl")
        if not os.path.exists(results_path):
            st.error("⚠️ Resultados Industrial não encontrados. Execute `python main.py` (DOMAIN_ACTIVE='INDUSTRIAL')")
            st.stop()
        with open(results_path, "rb") as f:
            return pickle.load(f)


def plotly_layout(title="", height=500, domain="SPACE"):
    """Layout padrão para gráficos Plotly com tema dark."""
    theme = config.THEME if domain == "SPACE" else config.THEME_INDUSTRIAL
    return dict(
        title=dict(text=title, font=dict(size=16, color=theme["text"])),
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["panel_bg"],
        font=dict(color=theme["text"], size=12),
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
        yaxis=dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
        legend=dict(
            bgcolor=theme["panel_bg"],
            bordercolor=theme["grid"],
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


def render_ai_analysis(domain, data, selected_idx=0):
    """Gera um relatório descritivo baseado na análise da IA."""
    st.markdown('<h2 class="section-title">🤖 Relatório de Análise da IA</h2>', unsafe_allow_html=True)
    
    # CSS para garantir que as tags HTML herdadas do markdown fiquem brancas
    st.markdown("""
    <style>
        .report-box h3, .report-box p, .report-box li, .report-box div { color: white !important; }
        .report-box { 
            background-color: rgba(255, 255, 255, 0.05); 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        if domain == "SPACE":
            summary = data["model_summary"]
            target_result = summary["target"]
            pca_var = data["pca_variance"]
            tc = target_result["cluster"]
            hyp = config.CLUSTER_HYPOTHESES.get(tc, {})
            target_idx = data["names"].index(str(data["target_star"]["kic_id"]))
            target_feats = {name.lower(): val for name, val in zip(data["feature_names"], data["feature_matrix"][target_idx])}

            col_rep1, col_rep2 = st.columns(2)
            with col_rep1:
                st.markdown(f"""
                <div class="report-box">
                    <h3 style='margin-top:0'>{config.TARGET_STAR['kic_id']} — {config.TARGET_STAR['nome']}</h3>
                    <p><b>Descrição:</b> {config.TARGET_STAR['descricao']}</p>
                    <hr style='border-color: rgba(255,255,255,0.1)'>
                    <div class="report-table">
                    
| Feature | Valor |
|---------|-------|
| Desvio Padrão (std) | `{target_feats.get('std', 0):.6f}` |
| Amplitude (range) | `{target_feats.get('range', 0):.6f}` |
| Assimetria (skewness) | `{target_feats.get('skewness', 0):.6f}` |
| MAD | `{target_feats.get('mad', 0):.6f}` |
| Curtose (kurtosis) | `{target_feats.get('kurtosis', 0):.6f}` |

                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_rep2:
                st.markdown(f"""
                <div class="report-box">
                    <h3 style='margin-top:0'>Resultado dos Modelos</h3>
                    <ul>
                        <li><b>Isolation Forest:</b> {'🔴 ANOMALIA' if target_result['is_anomaly'] else '🟢 NORMAL'}</li>
                        <li><b>Cluster K-Means:</b> C{tc+1} — {hyp.get('nome', '?')}</li>
                        <li><b>Silhouette Score:</b> `{summary['silhouette_score']:.3f}`</li>
                    </ul>
                    <hr style='border-color: rgba(255,255,255,0.1)'>
                    <h3 style='font-size: 1.2rem'>Hipótese Principal</h3>
                    <blockquote style='border-left: 3px solid gold; padding-left: 10px; font-style: italic; color: #ddd;'>
                        {hyp.get('hipotese', 'N/A')}
                    </blockquote>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            # --- DIAGNÓSTICO INDUSTRIAL AVANÇADO ---
            df = pd.DataFrame(data["data"])
            feature_cols = data["feature_cols"]
            analytics = data.get("analytics")
            model = data.get("model")
            
            asset_data = df.iloc[selected_idx]
            is_anomaly = analytics["if_preds"][selected_idx] == -1 if analytics else False
            score = analytics["if_scores"][selected_idx] if analytics else 0
            
            diagnostic = "N/A"
            confidence = 0
            top_drivers = []
            
            if model:
                X_sample = asset_data[feature_cols].values.reshape(1, -1)
                pred_res = model.predict(X_sample)
                diagnostic = pred_res["diagnostic"]
                confidence = pred_res["diagnostic_confidence"]
                shap_vals = np.abs(pred_res["shap_values"])
                top_indices = np.argsort(shap_vals)[-3:][::-1]
                top_drivers = [feature_cols[i] for i in top_indices]

            col_rep1, col_rep2 = st.columns(2)

            with col_rep1:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(f"### Ativo ID: {selected_idx} — Perfil Crítico")
                st.markdown(f"**Status:** {'🔴 ALERTA DE FALHA' if is_anomaly else '🟢 OPERAÇÃO NORMAL'}")
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
                st.markdown(f"""
| Sensor | Leitura Atual |
|---------|-------|
| Temperatura Ar | `{asset_data['Air temperature [K]']:.2f} K` |
| Torque | `{asset_data['Torque [Nm]']:.2f} Nm` |
| Rotação | `{asset_data['Rotational speed [rpm]']:.0f} RPM` |
| Desgaste Ferramenta | `{asset_data['Tool wear [min]']:.0f} min` |
| Diferença Térmica | `{asset_data['Temp_Diff']:.2f} K` |
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_rep2:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown("### Diagnóstico Sugerido (IA)")
                st.markdown(f"""
- **Tipo de Falha Histórica:** {diagnostic}
- **Confiança do Modelo:** `{confidence:.1%}`
- **Score de Anomalia (IF):** `{score:.4f}`
                """)
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
                st.markdown("#### Análise de Causa Raiz (XAI)")
                st.markdown("Os principais vetores de instabilidade são:")
                st.markdown(f"""
1. **{top_drivers[0] if len(top_drivers)>0 else 'N/A'}**
2. **{top_drivers[1] if len(top_drivers)>1 else 'N/A'}**
3. **{top_drivers[2] if len(top_drivers)>2 else 'N/A'}**
                """)
                st.info(f"**Recomendação:** { 'Priorizar manutenção imediata' if is_anomaly else 'Manter cronograma padrão'}")
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CONTROLE DE DOMÍNIO (SIDEBAR)
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌐 Domínio de Observabilidade")
    active_domain = st.selectbox(
        "Selecione o ambiente:",
        options=["SPACE", "INDUSTRIAL"],
        index=0 if config.DOMAIN_ACTIVE == "SPACE" else 1
    )
    st.markdown("---")

data = load_results(active_domain)

if active_domain == "SPACE":
    # --- Lógica Space (Original) ---
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
    feature_names = data["feature_names"]
    
    with st.sidebar:
        st.markdown(f"**Estrela Alvo:** {target_star['kic_id']}")
        st.markdown(f"**Nome:** {target_star['nome']}")
        st.markdown("---")
        st.markdown("### 📊 Dataset")
        st.markdown(f"- Total: {summary['n_total']}")
        st.markdown(f"- Anomalias: {summary['n_anomalies']}")
        show_ref_curves = st.checkbox("Mostrar referências", value=True)

    st.markdown("""
    # 🌟 Detecção de Anomalias Estelares
    ### Telescópio Kepler / NASA — Isolation Forest + K-Means
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1: render_metric_card("Status", "ANOMALIA ⚠️" if target_result["is_anomaly"] else "NORMAL ✓", "anomaly" if target_result["is_anomaly"] else "normal")
    with col2: render_metric_card("Score IF", f"{target_result['if_score']:.4f}")
    with col3: render_metric_card("Cluster", f"C{target_result['cluster'] + 1}") # +1 para ser 1-indexed
    with col4: render_metric_card("Silhouette", f"{summary['silhouette_score']:.3f}")

    # --- CHART 1: 📈 Curva de Luz (Scatter plot with markers) ---
    st.markdown('<h2 class="section-title">📈 Curva de Luz Interativa</h2>', unsafe_allow_html=True)
    
    # Seletor de Estrela
    star_names = [f"ALVO ({target_star['kic_id']})"] + [f"REF: {s['kic_id']} ({s['nome']})" for s in config.REFERENCE_STARS['estaveis'] + config.REFERENCE_STARS['variaveis'] + config.REFERENCE_STARS['anomalas']]
    selected_name = st.selectbox("Selecione a Estrela para Visualizar:", star_names)
    
    # Resolver estrela selecionada
    if "ALVO" in selected_name:
        current_lc = target_proc
        star_label = f"Alvo ({target_star['kic_id']})"
    else:
        # Procurar nas referências carregadas
        sel_kic = selected_name.split(":")[1].split("(")[0].strip()
        ref_found = next((s for s in ref_stars if s['kic_id'] == sel_kic), None)
        current_lc = ref_found if ref_found else target_proc
        star_label = selected_name

    fig_lc = go.Figure()
    
    # 1. Linha de Referência (fluxo = 1.0)
    fig_lc.add_hline(y=1.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", annotation_text="Referência (Fluxo = 1.0)")
    
    # 2. Curvas de Referência Opcionais (fundo)
    if show_ref_curves and ref_stars:
        for i, ref in enumerate(ref_stars[:8]): # Mostrar algumas de fundo
             fig_lc.add_trace(go.Scattergl(
                x=ref["time"], y=ref["flux"],
                mode="lines",
                line=dict(width=0.5, color="rgba(255, 255, 255, 0.08)"),
                hoverinfo="skip",
                showlegend=False
            ))
            
    # 3. Curva Selecionada (Principal)
    fig_lc.add_trace(go.Scattergl(
        x=current_lc["time"], y=current_lc["flux"],
        mode="markers",
        marker=dict(size=4, color=config.THEME["accent1"] if "ALVO" in selected_name else "rgba(255,255,255,0.6)"),
        name=star_label,
        hovertemplate="Tempo: %{x}<br>Fluxo: %{y:.6f}<extra></extra>"
    ))
    
    fig_lc.update_layout(**plotly_layout("Fluxo Normalizado vs Tempo (BJD)", domain="SPACE"))
    st.plotly_chart(fig_lc, width="stretch")

    # --- CHARTS 2 & 3: PCA Space ---
    st.markdown('<h2 class="section-title">🔍 Espaço PCA (Projeção Latente)</h2>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    
    target_idx = names.index(str(target_star["kic_id"]))
    target_pca = pca_coords[target_idx]
    
    # helper for highlighting target
    def add_target_to_pca(fig, coords):
        fig.add_trace(go.Scatter(
            x=[coords[0]], y=[coords[1]],
            mode="markers",
            marker=dict(size=15, color="white", symbol="star", line=dict(width=1, color="gold")),
            name="Estrela Alvo",
            showlegend=True
        ))

    with col_p1:
        # Chart 2: PCA com Isolation Forest (Red/Blue)
        # 1 = normal, -1 = anomalia
        colors_if = ["#4444ff" if p == 1 else "#ff4444" for p in if_preds]
        fig_if = px.scatter(
            x=pca_coords[:,0], y=pca_coords[:,1],
            color=if_preds.astype(str),
            color_discrete_map={"1": "#4444ff", "-1": "#ff4444"},
            labels={"x": f"PC1 ({pca_var[0]*100:.1f}%)", "y": f"PC2 ({pca_var[1]*100:.1f}%)"},
            title="Isolation Forest (Normal vs Anômalo)"
        )
        add_target_to_pca(fig_if, target_pca)
        fig_if.update_layout(**plotly_layout("Detecção Outliers", domain="SPACE"))
        st.plotly_chart(fig_if, width="stretch")
        
    with col_p2:
        # Chart 3: PCA com K-Means
        fig_km = px.scatter(
            x=pca_coords[:,0], y=pca_coords[:,1],
            color=cluster_labels.astype(str),
            labels={"x": "PC1", "y": "PC2"},
            title="K-Means Clusters"
        )
        add_target_to_pca(fig_km, target_pca)
        fig_km.update_layout(**plotly_layout("Agrupamento Morfológico", domain="SPACE"))
        st.plotly_chart(fig_km, width="stretch")

    # --- CHART 4 & 5: Scores & Radar ---
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        # Chart 4: Histograma de Scores
        st.markdown('<h3 class="section-title">📊 Histograma de Scores</h3>', unsafe_allow_html=True)
        fig_hist = px.histogram(if_scores, nbins=50, color_discrete_sequence=[config.THEME["accent1"]])
        
        # Vertical lines
        target_score = if_scores[target_idx]
        threshold_5 = np.percentile(if_scores, 5)
        
        fig_hist.add_vline(x=target_score, line_dash="dash", line_color="white", annotation_text=f"Alvo ({target_score:.2f})")
        fig_hist.add_vline(x=threshold_5, line_dash="dot", line_color="rgba(255,0,0,0.5)", annotation_text="Limiar 5%")
        
        fig_hist.update_layout(**plotly_layout("Distribuição de Anomalias", height=400, domain="SPACE"))
        st.plotly_chart(fig_hist, width="stretch")
        
    with col_s2:
        # Chart 5: Radar Chart (Perfil de Features)
        st.markdown('<h3 class="section-title">📊 Radar de Assinatura</h3>', unsafe_allow_html=True)
        
        # Normalização manual [0, 1] para o Radar
        f_min = feature_matrix.min(axis=0)
        f_max = feature_matrix.max(axis=0)
        norm_matrix = (feature_matrix - f_min) / (f_max - f_min + 1e-9)
        
        target_norm = norm_matrix[target_idx]
        
        fig_radar = go.Figure()
        
        # Plot clusters averages (translucency)
        for c in range(int(cluster_labels.max() + 1)):
            c_mask = cluster_labels == c
            c_avg = norm_matrix[c_mask].mean(axis=0)
            fig_radar.add_trace(go.Scatterpolar(
                r=c_avg, theta=feature_names,
                fill='toself', name=f'Cluster {c+1}',
                opacity=0.3
            ))
            
        # Target Highlight
        fig_radar.add_trace(go.Scatterpolar(
            r=target_norm, theta=feature_names,
            fill='none', name='ESTRELA ALVO',
            line=dict(color="gold", width=2)
        ))
        
        fig_radar.update_layout(polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor="rgba(0,0,0,0)"
        ), **plotly_layout("Perfil Morfológico", height=400, domain="SPACE"))
        st.plotly_chart(fig_radar, width="stretch")

    # --- CHART 6: 🌌 Espaço 3D ---
    st.markdown('<h2 class="section-title">🌌 Espaço de Atributos 3D</h2>', unsafe_allow_html=True)
    
    # Localizar indices das features (Std, Range, Skewness)
    try:
        idx_std = feature_names.index("Std")
        idx_range = feature_names.index("Range")
        idx_skew = feature_names.index("Skewness")
    except:
        idx_std, idx_range, idx_skew = 0, 1, 2 # Fallback
        
    fig_3d = px.scatter_3d(
        x=feature_matrix[:, idx_std],
        y=feature_matrix[:, idx_range],
        z=feature_matrix[:, idx_skew],
        color=cluster_labels.astype(str),
        labels={"x": "Std", "y": "Range", "z": "Skewness"},
        opacity=0.6
    )
    # Highlight Target
    fig_3d.add_trace(go.Scatter3d(
        x=[feature_matrix[target_idx, idx_std]],
        y=[feature_matrix[target_idx, idx_range]],
        z=[feature_matrix[target_idx, idx_skew]],
        mode="markers",
        marker=dict(size=10, color="gold", symbol="diamond"),
        name="Alvo (Destaque)"
    ))
    
    fig_3d.update_layout(**plotly_layout("Projeção Tridimensional", height=600, domain="SPACE"))
    st.plotly_chart(fig_3d, width="stretch")

    render_ai_analysis("SPACE", data)

else:
    # --- Lógica Industrial (Camaçari) ---
    df = pd.DataFrame(data["data"])
    summary = data["results_summary"]
    feature_cols = data["feature_cols"]
    
    # Analytics data (PCA, Clusters, IF)
    analytics = data.get("analytics")
    pca_coords = np.array(analytics["pca_coords"]) if analytics else None
    if_preds = np.array(analytics["if_preds"]) if analytics else None
    if_scores = np.array(analytics["if_scores"]) if analytics else None
    cluster_labels = np.array(analytics["cluster_labels"]) if analytics else None
    pca_var = summary.get("pca_variance", [0, 0])

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {config.THEME_INDUSTRIAL['bg']}; }}
        h1, h2, h3, h4, h5, h6, .section-title {{ color: {config.THEME_INDUSTRIAL['text']} !important; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    # 🏭 Centro de Controle Camaçari
    ### Manutenção Preditiva I4.0 — Unidade Industrial Polo
    """)

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: render_metric_card("Disponibilidade", f"{100 - (summary['anomaly_rate']*100):.1f}%", "normal")
    with col2: render_metric_card("Frequência de Falhas", f"{summary['anomaly_rate']*100:.1f}%", "anomaly")
    with col4: render_metric_card("Registros Analisados", summary["n_total"], "")
    with col3: 
        status_plant = "CRÍTICO" if summary['anomaly_rate'] > 0.05 else "NOMINAL"
        render_metric_card("Status Operacional", status_plant, "anomaly" if status_plant == "CRÍTICO" else "normal")

    st.markdown("---")
    
    # --- Interatividade Industrial ---
    st.sidebar.markdown("### 🔧 Configurações do Ativo")
    selected_idx = st.sidebar.number_input("Selecionar Registro (ID)", 0, len(df)-1, 0)
    selected_feature = st.sidebar.selectbox("Variável de Telemetria", feature_cols)
    
    # --- CHART 1: Telemetria Interativa ---
    st.markdown(f'<h2 class="section-title">⚙️ Telemetria em Tempo Real: {selected_feature}</h2>', unsafe_allow_html=True)
    
    # Mostrar janela de 500 pontos ao redor do selecionado para performance
    window = 250
    start_w = max(0, selected_idx - window)
    end_w = min(len(df), selected_idx + window)
    df_window = df.iloc[start_w:end_w]
    
    fig_telemetry = px.line(df_window, x=df_window.index, y=selected_feature, 
                            color_discrete_sequence=[config.THEME_INDUSTRIAL["accent1"]])
    # Highlight selected point
    fig_telemetry.add_trace(go.Scatter(
        x=[selected_idx], y=[df.at[selected_idx, selected_feature]],
        mode="markers", marker=dict(size=12, color="white", symbol="asterisk"),
        name="Ativo Selecionado"
    ))
    fig_telemetry.update_layout(**plotly_layout(f"Tendência de {selected_feature}", domain="INDUSTRIAL"))
    st.plotly_chart(fig_telemetry, width="stretch")

    # --- CHARTS 2 & 3: PCA Space ---
    if pca_coords is not None:
        st.markdown('<h2 class="section-title">🔍 Monitoramento Latente (PCA)</h2>', unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)
        
        target_pca = pca_coords[selected_idx]
        
        def add_target_ind(fig, coords):
            fig.add_trace(go.Scatter(
                x=[coords[0]], y=[coords[1]],
                mode="markers", marker=dict(size=15, color="white", symbol="diamond", line=dict(width=1, color="orange")),
                name="Foco de Análise"
            ))

        with col_p1:
            fig_if = px.scatter(
                x=pca_coords[:,0], y=pca_coords[:,1],
                color=if_preds.astype(str),
                color_discrete_map={"1": "#4444ff", "-1": "#ff4444"},
                labels={"x": f"PC1 ({pca_var[0]*100:.1f}%)", "y": f"PC2 ({pca_var[1]*100:.1f}%)"},
                title="Detecção de Anomalias (Isolation Forest)"
            )
            add_target_ind(fig_if, target_pca)
            fig_if.update_layout(**plotly_layout("Zonas de Risco", domain="INDUSTRIAL"))
            st.plotly_chart(fig_if, width="stretch")
            
        with col_p2:
            # Colorir por falha real se disponível
            fail_labels = df[config.INDUSTRIAL_PARAMS["target"]].astype(str)
            fig_faults = px.scatter(
                x=pca_coords[:,0], y=pca_coords[:,1],
                color=fail_labels,
                color_discrete_map={"0": "rgba(100,100,100,0.3)", "1": "#ffcc00"},
                labels={"x": "PC1", "y": "PC2"},
                title="Sinalização de Falhas Reais"
            )
            add_target_ind(fig_faults, target_pca)
            fig_faults.update_layout(**plotly_layout("Correlação de Falhas", domain="INDUSTRIAL"))
            st.plotly_chart(fig_faults, width="stretch")

    # --- CHART 4 & 5: Scores & Radar ---
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown('<h3 class="section-title">📊 Perfil de Risco Operacional</h3>', unsafe_allow_html=True)
        fig_hist = px.histogram(if_scores, nbins=50, color_discrete_sequence=[config.THEME_INDUSTRIAL["accent3"]])
        fig_hist.add_vline(x=if_scores[selected_idx], line_dash="dash", line_color="white", annotation_text="Ativo Atual")
        fig_hist.update_layout(**plotly_layout("Score de Anomalia (Outlier)", height=400, domain="INDUSTRIAL"))
        st.plotly_chart(fig_hist, width="stretch")
        
    with col_s2:
        st.markdown('<h3 class="section-title">📊 Assinatura de Sensores (Radar)</h3>', unsafe_allow_html=True)
        
        # Radar comparativo: Ativo vs Média Global
        f_vals = df.loc[selected_idx, feature_cols].values
        f_avg = df[feature_cols].mean().values
        
        # Normalização simples para o Radar
        f_min = df[feature_cols].min().values
        f_max = df[feature_cols].max().values
        norm_val = (f_vals - f_min) / (f_max - f_min + 1e-9)
        norm_avg = (f_avg - f_min) / (f_max - f_min + 1e-9)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=norm_avg, theta=feature_cols, fill='toself', name='Média Planta', opacity=0.3))
        fig_radar.add_trace(go.Scatterpolar(r=norm_val, theta=feature_cols, fill='toself', name='Ativo Selecionado', line=dict(color="orange")))
        
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                               **plotly_layout("Comparativo de Sensores", height=400, domain="INDUSTRIAL"))
        st.plotly_chart(fig_radar, width="stretch")

    # --- CHART 6: 3D Operational Space ---
    st.markdown('<h2 class="section-title">🚀 Espaço Operacional 3D (Sensores Críticos)</h2>', unsafe_allow_html=True)
    
    # Usar 3 features principais: Torque, Speed, Temp_Diff
    f3d_x, f3d_y, f3d_z = "Torque [Nm]", "Rotational speed [rpm]", "Temp_Diff"
    
    fig_3d = px.scatter_3d(
        df.iloc[::5], # Decimação para 3D fluído
        x=f3d_x, y=f3d_y, z=f3d_z,
        color=if_preds[::5].astype(str) if if_preds is not None else None,
        color_discrete_map={"1": "rgba(40,40,255,0.4)", "-1": "#ff3333"},
        opacity=0.6,
        title="Torque vs Velocidade vs Delta T"
    )
    fig_3d.update_layout(**plotly_layout("Dinâmica Multidimensional", height=600, domain="INDUSTRIAL"))
    st.plotly_chart(fig_3d, width="stretch")

    render_ai_analysis("INDUSTRIAL", data, selected_idx)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    f"<center style='opacity:0.5'>Dual-Domain Observability — TCC — "
    f"NASA/Kepler & AI4I Industrial — (Domínio Ativo: {active_domain})</center>",
    unsafe_allow_html=True,
)
