import pytest
import numpy as np
import pandas as pd
from pipeline.feature_engineer import PhotometricFeatureExtractor, IndustrialFeatureEngineer

def test_photometric_extractor_correct_features(feature_vector_names):
    """Verifica se as 6 features fotométricas são extraídas corretamente."""
    flux = np.random.normal(1.0, 0.05, 1000)
    extractor = PhotometricFeatureExtractor()
    
    features = extractor.extract(flux)
    
    assert len(features) == 6
    for name in feature_vector_names:
        assert name in features
        assert isinstance(features[name], float)

def test_photometric_skewness_negative():
    """Testa se a assimetria detecta quedas profundas."""
    flux = np.ones(100)
    # Adicionar uma queda profunda
    flux[40:50] = 0.5
    
    extractor = PhotometricFeatureExtractor()
    features = extractor.extract(flux)
    
    # Quedas profundas devem gerar assimetria negativa (muitos pontos abaixo da média)
    # Na verdade, se a maioria está em 1 e uns poucos em 0.5, a cauda está na esquerda.
    assert features["skewness"] < 0

def test_industrial_feature_engineer_calculations(mock_industrial_df):
    """Verifica os cálculos de Delta T e Potência."""
    eng = IndustrialFeatureEngineer()
    df_feat = eng.transform(mock_industrial_df)
    
    assert "Temp_Diff" in df_feat.columns
    assert "Power" in df_feat.columns
    
    # Verificar cálculo simples: Temp_Diff = Process - Air
    # 308.6 - 298.1 = 10.5
    assert pytest.approx(df_feat.iloc[0]["Temp_Diff"]) == 10.5
    # Power = RPM * (2*pi/60) * Torque
    # 1551 * 0.1047 * 42.8 ~ 6951
    assert df_feat.iloc[0]["Power"] > 0
