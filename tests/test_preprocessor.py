import pytest
import numpy as np
import pandas as pd
from pipeline.preprocessor import LightCurvePreprocessor, IndustrialPreprocessor

def test_lightcurve_preprocessor_remove_nans():
    """Verifica se Remove NaNs."""
    time = np.array([1, 2, 3, 4], dtype=float)
    flux = np.array([1, np.nan, 3, np.inf], dtype=float)
    pre = LightCurvePreprocessor()
    
    t_clean, f_clean = pre.remove_nans(time, flux)
    
    assert len(t_clean) == 2
    assert 2 not in t_clean
    assert np.all(np.isfinite(f_clean))

def test_light_curve_preprocessor_sigma_clip():
    """Verifica se Sigma Clipping remove outliers extremos."""
    time = np.linspace(0, 10, 100)
    flux = np.ones(100)
    # Adicionar outlier extremo (10 sigmas se o resto for constante? No, std é 0... 
    # vams colocar ruído + 1 outlier)
    flux += np.random.normal(0, 0.01, 100)
    flux[50] = 5.0 # Outlier gigante
    
    pre = LightCurvePreprocessor(outlier_sigma=3.0)
    t_clean, f_clean = pre.sigma_clip(time, flux)
    
    assert len(f_clean) < 100
    assert 5.0 not in f_clean

def test_industrial_preprocessor_cleans_columns(mock_industrial_df):
    """Verifica se remove colunas desnecessárias e mapeia tipos."""
    pre = IndustrialPreprocessor()
    df_clean = pre.process(mock_industrial_df)
    
    assert "UDI" not in df_clean.columns
    assert "Product ID" not in df_clean.columns
    assert "Type" in df_clean.columns
    # Verifica mapeamento ('M' -> 1)
    assert df_clean.iloc[0]["Type"] == 1
