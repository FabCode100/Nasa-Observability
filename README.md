# 🌌 🔭 Universal Observability: From Space to Industry

![universal_observability_banner](https://raw.githubusercontent.com/username/repo/main/banner.png) 
*(Nota: O design visual deste projeto foi atualizado para suportar observabilidade de sistemas críticos em múltiplos domínios, com foco em alta fidelidade e interpretabilidade)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NasaData](https://img.shields.io/badge/Data-Kepler_Mission-orange?logo=nasa&logoColor=white)](https://archive.stsci.edu/kepler/)
[![IndustrialData](https://img.shields.io/badge/Data-Predictive_Maintenance-red?logo=factory&logoColor=white)](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
[![ML](https://img.shields.io/badge/AI-Explainable_SHAP-green)](https://shap.readthedocs.io/)

## 🛰️ Visão Geral do Projeto
Este **Trabalho de Conclusão de Curso (TCC)** propõe uma infraestrutura de **Observabilidade Universal**. O sistema prova que as mesmas bases matemáticas (PCA, Isolation Forest, K-Means) utilizadas para detectar anomalias em estrelas distantes (Telescópio Kepler) podem ser aplicadas com alta fidelidade na **Manutenção Preditiva Industrial**.

O projeto opera em dois hubs especializados:
1.  **🌌 Stellar Hub**: Detecção de anomalias fotométricas em dados da NASA (Kepler).
2.  **🏭 Camaçari Hub**: Monitoramento de ativos industriais com diagnóstico de falhas em tempo real e XAI.

---

## 🚀 Centro de Comando (Dashboard 6-Chart Suite)
O painel foi modernizado para o padrão acadêmico/profissional (NASA Design System), apresentando uma suíte de 6 visualizações críticas:

- **⚙️ Telemetria Interativa**: Análise temporal de sensores com janela deslizante de alta performance.
- **🔍 Espaço Latente (PCA Anomaly)**: Projeção 2D das coordenadas principais para identificação de clusters de risco.
- **🔍 Diagnóstico Latente (PCA Clustering)**: Agrupamento não supervisionado de comportamentos operacionais (K-Means).
- **📊 Sensores Signature (Radar)**: Comparativo da "impressão digital" do ativo atual em relação à média estável da planta/universo.
- **📉 Perfil de Score (IF)**: Histograma da distribuição estatística de anomalias em todo o dataset.
- **🧊 Espaço Operacional 3D**: Visualização volumétrica (Ex: Torque x RPM x Temp) para identificação de zonas de colapso.

---

## 🧠 Inteligência e Explicabilidade (XAI)
O núcleo do projeto utiliza **Explainable AI (SHAP)** para transformar predições de "caixa-preta" em diagnósticos acionáveis:
- **Relatório Automatizado**: O sistema gera um laudo técnico para cada ativo/estrela.
- **Root Cause Analysis**: Identificação dos 3 principais vetores (sensores ou métricas fotométricas) que justificam o alerta de anomalia.

---

## 🛠️ Instalação e Operação

### 1. Preparar o Ambiente
```bash
git clone https://github.com/seu-usuario/Universal-Observability.git
cd Universal-Observability
pip install -r requirements.txt
```

### 2. Executar o Ciclo de Dados
```bash
# Processe os dados (baixa dataset, treina modelos e gera analytics)
# Agora com suporte a download paralelo (n_jobs=4)
python main.py

# Lance a estação de visualização
streamlit run dashboard.py
```

### 3. 🐳 Implantação com Docker (Recomendado)
Para uma experiência "plug-and-play" sem configurar ambientes locais:
```bash
# 1. Subir container (Dashboard + Ambiente)
docker-compose up --build

# 2. (Opcional) Rodar o pipeline dentro do Docker se quiser atualizar dados:
docker exec -it universal-observability python main.py
```
O painel estará disponível em: `http://localhost:8501`

### 4. Configurar Domínio
Edite `config.py` para alternar entre os ambientes:
```python
DOMAIN_ACTIVE = 'INDUSTRIAL'  # Opções: 'SPACE', 'INDUSTRIAL'
```

---

## 📂 Arquitetura do Pipeline
```bash
├── 📁 pipeline/         # Lógica modular
│   ├── data_collector.py  # Acesso a APIs NASA e UCI
│   ├── preprocessor.py    # Limpeza e Harmonização
│   ├── feature_engineer.py # Métricas de domínio (ex: ΔT, Mechanical Power)
│   └── models.py          # Motores IA (AnomalyDetector & PredictiveModel)
├── 📄 main.py           # Orquestrador Batch
├── 📄 dashboard.py      # Estação de Visualização Streamlit
├── 📄 config.py         # Centro de Variáveis Globais
├── 📄 defense_argument.md # Argumentação para banca
└── 📁 tests/           # Suíte de Testes Unitários (Pytest)
```

---

## ✅ Qualidade e Testes

O projeto utiliza **Pytest** para garantir a integridade matemática e funcional do pipeline.

### 1. Rodar Testes Localmente
```bash
pytest --cov=pipeline tests/
```

### 2. Logs Modernos (Loguru)
Os logs do sistema agora utilizam o framework **Loguru**, oferecendo:
- **Cores**: Melhor identificação de níveis (INFO, WARNING, ERROR) no terminal.
- **Rotação**: Logs arquivos são rotacionados quando atingem 10MB.
- **Retenção**: Mantemos os últimos 10 dias de execução compactados em `.zip`.
- **Localização**: `/logs/pipeline_YYYY-MM-DD.log`.

### 3. Integração Contínua (CI)
Toda mudança é validada via **GitHub Actions**, que executa:
- Testes unitários com cobertura.
- Verificação de Linting (Flake8).
- Validação do build do Docker.

---

<p align="center">
  <i>"Das estrelas para o chão de fábrica — Observabilidade sem fronteiras."</i>
</p>
