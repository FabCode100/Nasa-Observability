"""
══════════════════════════════════════════════════════════════
TCC Astrofísica — Configuração Central
Detecção de Anomalias Estelares em Dados do Telescópio Kepler
══════════════════════════════════════════════════════════════
"""

import os

# ── Diretórios ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Estrela alvo ──────────────────────────────────────────
TARGET_STAR = {
    "kic_id": "KIC 8462852",
    "nome": "Estrela de Tabby",
    "quarter": 16,
    "author": "Kepler",
    "descricao": (
        "Estrela com quedas de brilho aperiódicas e assimétricas, "
        "candidata a ocultação por megaestrutura ou nuvem de cometas."
    ),
}

# ── Estrelas de referência (dados reais do Kepler) ────────
# Estas estrelas serão baixadas para servir de comparação real
REFERENCE_STARS = {
    # ── Estáveis (solar-like, baixa variabilidade) ────
    "estaveis": [
        {"kic_id": "KIC 6116048",  "nome": "Solar-like G",   "quarter": 16},
        {"kic_id": "KIC 3632418",  "nome": "Solar-like G/K", "quarter": 16},
        {"kic_id": "KIC 8694723",  "nome": "Subgiant G",     "quarter": 16},
        {"kic_id": "KIC 9955598",  "nome": "Main seq. G",    "quarter": 16},
        {"kic_id": "KIC 10963065", "nome": "Solar analog",   "quarter": 16},
    ],
    # ── Variáveis (pulsantes, exoplanetas gigantes) ───
    "variaveis": [
        {"kic_id": "KIC 11515377", "nome": "Pulsante δ Sct",     "quarter": 16},
        {"kic_id": "KIC 7548479",  "nome": "RR Lyrae candidata", "quarter": 16},
        {"kic_id": "KIC 5024327",  "nome": "Eclipsante",         "quarter": 16},
        {"kic_id": "KIC 10666592", "nome": "Trânsito gigante",   "quarter": 16},
        {"kic_id": "KIC 3861595",  "nome": "Variável longa",     "quarter": 16},
    ],
    # ── Anômalas (comportamento extremo) ──────────────
    "anomalas": [
        {"kic_id": "KIC 4150804",  "nome": "Heartbeat star",     "quarter": 16},
        {"kic_id": "KIC 12557548", "nome": "Desintegrating",     "quarter": 16},
        {"kic_id": "KIC 2856960",  "nome": "Triplo eclipsante",  "quarter": 16},
    ],
}

# ── Parâmetros de Pré-processamento ──────────────────────
PREPROCESS = {
    "outlier_sigma": 5,
    "flatten_window": 401,
}

# ── Parâmetros da População Simulada ─────────────────────
# Usada para complementar as estrelas reais
SIMULATED_POP = {
    "seed": 42,
    "n_estaveis":  500,
    "n_variaveis": 150,
    "n_extremas":  30,
    "ranges": {
        "estaveis":  {"std": (0.0003, 0.0015), "range": (0.001, 0.006), "skew": (-0.1, 0.1)},
        "variaveis": {"std": (0.002,  0.006),  "range": (0.01,  0.03),  "skew": (-0.8, 0.8)},
        "extremas":  {"std": (0.008,  0.018),  "range": (0.04,  0.12),  "skew": (-2.5, 2.5)},
    },
}

# ── Parâmetros dos Modelos ────────────────────────────────
ISOLATION_FOREST = {
    "n_estimators": 200,
    "contamination": 0.051,
    "random_state": 42,
    "n_jobs": -1,
}

KMEANS = {
    "n_clusters": 3,
    "n_init": 20,
    "random_state": 42,
}

PCA_PARAMS = {
    "n_components": 2,
    "random_state": 42,
}

# ── Hipóteses por Cluster ────────────────────────────────
CLUSTER_HYPOTHESES = {
    0: {
        "nome": "Anãs G/K Estáveis",
        "cor": "#2271B3",
        "descricao": "Estrelas semelhantes ao Sol sem trânsito expressivo.",
        "hipotese": (
            "Candidatas ideais para busca de planetas tipo-Terra "
            "via fotometria de precisão."
        ),
    },
    1: {
        "nome": "Variáveis / Gigantes Gasosos",
        "cor": "#3B8228",
        "descricao": "Pulsantes (Cepheids, RR Lyrae) ou com trânsito de Júpiter quente.",
        "hipotese": (
            "A variabilidade periódica sugere pulsação intrínseca "
            "ou trânsito de exoplaneta gigante."
        ),
    },
    2: {
        "nome": "Anomalias Fotométricas Extremas",
        "cor": "#A32D2D",
        "descricao": "Quedas assimétricas, aperiódicas e de alta amplitude.",
        "hipotese": (
            "Nuvem de cometas fragmentados, disco de poeira pós-colisão, "
            "binária eclipsante exótica ou — hipótese mais especulativa — "
            "estrutura artificial circunstelar (Esfera de Dyson parcial)."
        ),
    },
}

# ── Tema Visual (Dark Astronomy) ─────────────────────────
THEME = {
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
    "text":     "#c9d1d9",
    "grid":     "#21262d",
    "accent1":  "#58a6ff",  # Azul (normais / estáveis)
    "accent2":  "#3fb950",  # Verde (variáveis)
    "accent3":  "#ff7b72",  # Vermelho (anomalias)
    "target":   "#ffa657",  # Laranja (estrela alvo)
    "cluster_colors": ["#58a6ff", "#3fb950", "#ff7b72"],
    "cluster_names":  ["C0: Estáveis", "C1: Variáveis", "C2: Extremas"],
}
