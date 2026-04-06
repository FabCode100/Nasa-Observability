import pytest
import numpy as np
from pipeline.models import AnomalyDetector, PredictiveMaintenanceModel

def test_anomaly_detector_fit_predict():
    """Testa se o detector de anomalias treina e retorna o summary."""
    # Gerar 100 estrelas fakes com 6 features cada
    X = np.random.normal(0, 1, (100, 6))
    detector = AnomalyDetector()
    
    # 100 features normais + 1 anomalia extrema
    X[-1] = [10, 10, 10, 10, 10, 10]
    
    detector.fit(X)
    summary = detector.get_summary(target_index=-1)
    
    assert "target" in summary
    assert "n_total" in summary
    assert summary["n_total"] == 100
    # A anomalia deve ser detectada (-1) ou pelo menos estar no summary
    assert "is_anomaly" in summary["target"]

def test_predictive_maintenance_model(mock_industrial_df):
    """Testa o modelo industrial com SHAP e diagnóstico."""
    # Simular modelo pronto
    X = np.random.normal(300, 10, (10, 6)) # 6 features
    y = np.zeros(10)
    y_diag = np.zeros(10, dtype=int)
    
    model = PredictiveMaintenanceModel()
    model.fit(X, y, y_diag)
    
    # Prever para uma linha
    X_test = X[0:1]
    res = model.predict(X_test)
    
    assert "is_anomaly" in res
    assert "diagnostic" in res
    assert "shap_values" in res
    assert len(res["shap_values"]) == 6 # 6 features

def test_anomaly_detector_silhouette():
    """Verifica se o silhouette score é calculado (para K > 1)."""
    X = np.vstack([
        np.random.normal(0, 0.1, (20, 6)),
        np.random.normal(5, 0.1, (20, 6))
    ])
    
    detector = AnomalyDetector(km_params={"n_clusters": 2})
    detector.fit(X)
    
    assert detector.silhouette > 0.5 # Clusters bem separados
