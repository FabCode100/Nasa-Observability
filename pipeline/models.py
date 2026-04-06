"""
══════════════════════════════════════════════════════════════
Módulo: models.py
Modelos de detecção de anomalias (Isolation Forest, K-Means, PCA)
══════════════════════════════════════════════════════════════
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, Optional, List
import shap
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

# ... (rest of AnomalyDetector)
# ...

class PredictiveMaintenanceModel:
    """
    Modelo de Manutenção Preditiva com diagnóstico de falhas.
    
    Arquitetura:
    1. Isolation Forest: Detecção não-supervisionada de anomalias (anomalia vs normal).
    2. Random Forest: Classificação supervisionada dos modos de falha (diagnostic).
    3. SHAP: Explicação local para o motivo da falha.
    """

    def __init__(self, failure_modes: List[str] = None):
        self.failure_modes = failure_modes or ["TWF", "HDF", "PWF", "OSF", "RNF"]
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.04, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray, y_diagnostics: np.ndarray):
        """
        Treina o modelo de anomalia e o classificador de diagnóstico.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        
        # Treinar classificador para diagnóstico
        # Requer balanceamento (SMOTE) se disponível, pois as falhas são raras
        if SMOTE is not None:
             sm = SMOTE(random_state=42)
             X_sm, y_sm = sm.fit_resample(X_scaled, y_diagnostics)
             self.classifier.fit(X_sm, y_sm)
        else:
             self.classifier.fit(X_scaled, y_diagnostics)
             
        # Inicializar o SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.classifier)
        
        return self

    def predict(self, X: np.ndarray) -> Dict:
        """
        Retorna anomalia, diagnóstico e explicação local.
        """
        X_scaled = self.scaler.transform(X)
        is_anomaly = self.isolation_forest.predict(X_scaled) == -1
        
        # Diagnóstico (maior probabilidade de falha)
        diag_probs = self.classifier.predict_proba(X_scaled)[0]
        failure_idx = np.argmax(diag_probs)
        failure_mode = self.failure_modes[failure_idx] if failure_idx < len(self.failure_modes) else "N/A"
        
        # SHAP Values para a explicação
        shap_values = self.explainer.shap_values(X_scaled)[failure_idx]
        
        return {
            "is_anomaly": bool(is_anomaly[0]),
            "diagnostic": failure_mode,
            "diagnostic_confidence": float(diag_probs[failure_idx]),
            "shap_values": shap_values[0].tolist()
        }


class AnomalyDetector:
    """
    Encapsula o pipeline de detecção de anomalias estelares.

    Modelos:
        - StandardScaler: normalização das features
        - Isolation Forest: detecção de anomalias (1=normal, -1=anomalia)
        - K-Means: agrupamento em clusters
        - PCA: redução para 2D para visualização
    """

    def __init__(
        self,
        if_params: Optional[Dict] = None,
        km_params: Optional[Dict] = None,
        pca_params: Optional[Dict] = None,
    ):
        # Parâmetros padrão
        if_params = if_params or {
            "n_estimators": 200,
            "contamination": 0.051,
            "random_state": 42,
            "n_jobs": -1,
        }
        km_params = km_params or {
            "n_clusters": 3,
            "n_init": 20,
            "random_state": 42,
        }
        pca_params = pca_params or {
            "n_components": 2,
            "random_state": 42,
        }

        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(**if_params)
        self.kmeans = KMeans(**km_params)
        self.pca = PCA(**pca_params)

        # Resultados (preenchidos após fit)
        self.X_raw = None
        self.X_scaled = None
        self.X_pca = None
        self.if_predictions = None
        self.if_scores = None
        self.cluster_labels = None
        self.pca_variance = None
        self.silhouette = None

    def fit(self, X: np.ndarray):
        """
        Treina todos os modelos no dataset completo.

        Args:
            X: Matriz de features (N × D).
        """
        self.X_raw = X.copy()

        # 1. Normalização
        self.X_scaled = self.scaler.fit_transform(X)

        # 2. Isolation Forest
        self.if_predictions = self.isolation_forest.fit_predict(self.X_scaled)
        self.if_scores = self.isolation_forest.score_samples(self.X_scaled)

        # 3. K-Means
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)

        # 4. PCA
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.pca_variance = self.pca.explained_variance_ratio_

        # 5. Silhouette Score (qualidade do clustering)
        if len(np.unique(self.cluster_labels)) > 1:
            # BUGFIX: Para datasets grandes (> 2000), o cálculo da matriz de 
            # distância (N x N) causa MemoryError. Usar amostragem agressiva.
            if len(self.X_scaled) > 2000:
                indices = np.random.choice(len(self.X_scaled), 2000, replace=False)
                self.silhouette = silhouette_score(self.X_scaled[indices], self.cluster_labels[indices])
            else:
                self.silhouette = silhouette_score(self.X_scaled, self.cluster_labels)
        else:
            self.silhouette = 0.0

        return self

    def get_target_results(self, target_index: int = -1) -> Dict:
        """
        Retorna os resultados para a estrela alvo.

        Args:
            target_index: Índice da estrela alvo na matriz (-1 = última).

        Returns:
            Dict com predição, score, cluster, coordenadas PCA.
        """
        return {
            "if_prediction": int(self.if_predictions[target_index]),
            "if_score": float(self.if_scores[target_index]),
            "cluster": int(self.cluster_labels[target_index]),
            "pca_coords": self.X_pca[target_index].tolist(),
            "is_anomaly": bool(self.if_predictions[target_index] == -1),
        }

    def get_cluster_stats(self) -> Dict:
        """Estatísticas por cluster."""
        stats = {}
        for c in range(self.kmeans.n_clusters):
            mask = self.cluster_labels == c
            stats[c] = {
                "count": int(np.sum(mask)),
                "anomaly_count": int(np.sum(self.if_predictions[mask] == -1)),
                "mean_score": float(np.mean(self.if_scores[mask])),
                "pca_centroid": self.X_pca[mask].mean(axis=0).tolist(),
            }
        return stats

    def get_summary(self, target_index: int = -1) -> Dict:
        """Resumo completo dos resultados."""
        return {
            "target": self.get_target_results(target_index),
            "clusters": self.get_cluster_stats(),
            "pca_variance": self.pca_variance.tolist(),
            "silhouette_score": float(self.silhouette),
            "n_total": len(self.X_raw),
            "n_anomalies": int(np.sum(self.if_predictions == -1)),
            "anomaly_threshold": float(np.percentile(self.if_scores, 5)),
        }
