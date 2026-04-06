import pytest
import os
import pandas as pd
from unittest.mock import MagicMock, patch
from pipeline.data_collector import StellarDataCollector, IndustrialDataCollector

def test_industrial_collector_download(tmp_path, mock_industrial_df):
    """Testa o download e cache do dataset industrial."""
    cache_dir = tmp_path / "cache"
    collector = IndustrialDataCollector(cache_dir=str(cache_dir))
    
    # Mocking pd.read_csv to return our fixture
    with patch("pandas.read_csv", return_value=mock_industrial_df):
        df = collector.download_dataset("http://fake-url.com/data.csv")
        
        assert not df.empty
        assert len(df) == 3
        # Verificar se o arquivo foi salvo em cache
        assert os.path.exists(collector.csv_path)

@patch("pipeline.data_collector.StellarDataCollector._get_lightkurve")
def test_stellar_download_lightcurve(mock_get_lk, tmp_path):
    """Testa o download de curva de luz com mock do lightkurve."""
    cache_dir = tmp_path / "stellar_cache"
    collector = StellarDataCollector(cache_dir=str(cache_dir))
    
    # Configurar mock do lightkurve
    mock_lk = MagicMock()
    mock_get_lk.return_value = mock_lk
    
    # Mock do resultado da busca
    mock_search = MagicMock()
    mock_lk.search_lightcurve.return_value = mock_search
    mock_search.__len__.return_value = 1
    
    # Mock da curva de luz baixada
    mock_lc = MagicMock()
    mock_lc.time.value = [1, 2, 3]
    mock_lc.flux.value = [1.0, 0.9, 1.1]
    mock_search.download.return_value = mock_lc
    
    result = collector.download_lightcurve("KIC 123", quarter=16)
    
    assert result is not None
    assert result["kic_id"] == "KIC 123"
    assert len(result["flux"]) == 3
    assert os.path.exists(collector._cache_path("KIC 123", 16))

def test_stellar_simulation():
    """Verifica se a geração de estrelas simuladas funciona."""
    star = StellarDataCollector.generate_simulated_star(n_points=100, seed=42)
    
    assert "time" in star
    assert "flux" in star
    assert len(star["flux"]) == 100
    assert star["kic_id"] == "SIMULADA"
