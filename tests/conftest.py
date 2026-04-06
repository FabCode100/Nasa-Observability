import pytest
import numpy as np
import pandas as pd
from pipeline.data_collector import StellarDataCollector

@pytest.fixture
def mock_stellar_data():
    """Gera dados fakes de uma estrela para testes."""
    return {
        "kic_id": "KIC 8462852",
        "quarter": 16,
        "time": np.linspace(0, 10, 100),
        "flux": np.random.normal(1.0, 0.01, 100),
        "flux_err": np.random.normal(0.001, 0.0001, 100)
    }

@pytest.fixture
def mock_industrial_df():
    """Gera um dataframe fake do dataset AI4I 2020."""
    data = {
        "UDI": [1, 2, 3],
        "Product ID": ["M14860", "L47181", "L47182"],
        "Type": ["M", "L", "L"],
        "Process temperature [K]": [308.6, 308.7, 308.5],
        "Air temperature [K]": [298.1, 298.2, 298.3],
        "Rotational speed [rpm]": [1551, 1408, 1498],
        "Torque [Nm]": [42.8, 46.3, 49.4],
        "Tool wear [min]": [0, 3, 5],
        "Machine failure": [0, 0, 0],
        "TWF": [0, 0, 0],
        "HDF": [0, 0, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 0],
        "RNF": [0, 0, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def simulated_star():
    """Usa o gerador interno para criar uma estrela controlada."""
    return StellarDataCollector.generate_simulated_star(
        n_points=500, 
        seed=42
    )

@pytest.fixture
def feature_vector_names():
    """Lista das features extraídas."""
    from pipeline.feature_engineer import FEATURE_NAMES
    return FEATURE_NAMES
