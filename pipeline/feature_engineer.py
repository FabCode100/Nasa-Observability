"""
══════════════════════════════════════════════════════════════
Módulo: feature_engineer.py
Extração de features fotométricas de curvas de luz
══════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.stats import skew
from typing import Dict, List
import pandas as pd

# ... (rest of Stellar Feature Info)
# ...

INDUSTRIAL_FEATURE_INFO = {
    "Process temperature [K]": {"nome": "Temp. Processo", "unidade": "K"},
    "Air temperature [K]": {"nome": "Temp. Ar", "unidade": "K"},
    "Rotational speed [rpm]": {"nome": "Velocidade", "unidade": "RPM"},
    "Torque [Nm]": {"nome": "Torque", "unidade": "Nm"},
    "Tool wear [min]": {"nome": "Desgaste", "unidade": "min"},
    "Temp_Diff": {"nome": "ΔT (Process-Air)", "unidade": "K"},
    "Power": {"nome": "Potência Mecânica", "unidade": "W"},
}

class IndustrialFeatureEngineer:
    """
    Engenharia de features para o dataset de Manutenção Preditiva.
    Calcula indicadores físicos derivados:
    1. Temp_Diff: Diferença entre temperatura do processo e do ar.
    2. Power: Potência (Velocidade * Torque).
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features calculadas ao dataframe.
        """
        if df.empty:
            return df

        df_feat = df.copy()

        # 1. Diferença de temperatura
        if "Process temperature [K]" in df_feat.columns and "Air temperature [K]" in df_feat.columns:
            df_feat["Temp_Diff"] = (
                df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
            )

        # 2. Potência Mecânica (P = τ * ω)
        # ω (rad/s) = rpm * 2π / 60
        if "Rotational speed [rpm]" in df_feat.columns and "Torque [Nm]" in df_feat.columns:
            df_feat["Power"] = (
                df_feat["Rotational speed [rpm]"] * (2 * np.pi / 60) * df_feat["Torque [Nm]"]
            )

        return df_feat


# Nome e descrição de cada feature para documentação / dashboard
FEATURE_INFO = {
    "std": {
        "nome": "Desvio Padrão",
        "descricao": "Variabilidade total (ruído + trânsito)",
        "unidade": "fluxo relativo",
    },
    "range": {
        "nome": "Amplitude",
        "descricao": "Diferença entre máximo e mínimo (profundidade máxima da queda)",
        "unidade": "fluxo relativo",
    },
    "skewness": {
        "nome": "Assimetria (Skewness)",
        "descricao": "Assimetria temporal — valores negativos indicam mais quedas que picos",
        "unidade": "adimensional",
    },
    "mad": {
        "nome": "MAD",
        "descricao": "Desvio absoluto mediano — robusto a outliers",
        "unidade": "fluxo relativo",
    },
    "kurtosis": {
        "nome": "Curtose",
        "descricao": "Presença de caudas pesadas na distribuição do fluxo",
        "unidade": "adimensional",
    },
    "below_p2": {
        "nome": "Fração < P2",
        "descricao": "Frequência de pontos abaixo do percentil 2 (quedas extremas)",
        "unidade": "fração",
    },
}

FEATURE_NAMES = list(FEATURE_INFO.keys())


class PhotometricFeatureExtractor:
    """
    Extrai um vetor de 6 features fotométricas de uma curva de luz.

    Cada feature captura um aspecto físico diferente do comportamento
    estelar, permitindo a distinção entre estrelas estáveis, variáveis
    e anômalas.
    """

    @staticmethod
    def extract(flux: np.ndarray) -> Dict[str, float]:
        """
        Extrai features de um array de fluxo.

        Args:
            flux: Array 1D de fluxo normalizado.

        Returns:
            Dict com 6 features numéricas.
        """
        f = np.asarray(flux, dtype=float)
        f = f[np.isfinite(f)]

        if len(f) < 10:
            return {name: 0.0 for name in FEATURE_NAMES}

        med = np.median(f)
        std_val = np.std(f)

        return {
            "std":      float(std_val),
            "range":    float(np.max(f) - np.min(f)),
            "skewness": float(skew(f)),
            "mad":      float(np.median(np.abs(f - med))),
            "kurtosis": float(
                np.mean((f - np.mean(f)) ** 4) / (std_val ** 4 + 1e-10)
            ),
            "below_p2": float(np.mean(f < np.percentile(f, 2))),
        }

    def extract_from_star(self, star_data: Dict) -> Dict:
        """
        Extrai features de um dict de estrela (com chave 'flux').

        Retorna o dict original enriquecido com a chave 'features'.
        """
        features = self.extract(star_data["flux"])
        result = dict(star_data)
        result["features"] = features
        return result

    def extract_batch(self, star_list: List[Dict]) -> List[Dict]:
        """Extrai features de uma lista de estrelas."""
        results = []
        for star in star_list:
            enriched = self.extract_from_star(star)
            results.append(enriched)
        return results

    @staticmethod
    def features_to_vector(features: Dict[str, float]) -> np.ndarray:
        """Converte dict de features para vetor numpy na ordem canônica."""
        return np.array([features[name] for name in FEATURE_NAMES])

    @staticmethod
    def build_feature_matrix(star_list: List[Dict]) -> np.ndarray:
        """
        Constrói a matriz de features (N × 6) a partir de uma lista
        de estrelas com features extraídas.
        """
        vectors = []
        for star in star_list:
            feat = star.get("features", {})
            vec = np.array([feat.get(name, 0.0) for name in FEATURE_NAMES])
            vectors.append(vec)
        return np.array(vectors)


def generate_simulated_features(
    n: int,
    std_range,
    range_range,
    skew_range,
    seed: int = 42,
) -> np.ndarray:
    """
    Gera features simuladas para uma população estelar.

    Cada estrela simulada recebe 6 features com distribuições
    realistas baseadas nos intervalos fornecidos.

    Returns:
        Array (n, 6) de features simuladas.
    """
    rng = np.random.RandomState(seed)
    pop = []
    for _ in range(n):
        f_std = rng.uniform(*std_range)
        f_range = rng.uniform(*range_range) + f_std * 2
        f_skew = rng.uniform(*skew_range)
        f_mad = f_std * rng.uniform(0.6, 0.9)
        f_kurt = rng.uniform(2.5, 4.5) + abs(f_skew)
        f_below = rng.uniform(0.001, 0.005) * (1 + abs(f_skew))
        pop.append([f_std, f_range, f_skew, f_mad, f_kurt, f_below])
    return np.array(pop)
