"""
══════════════════════════════════════════════════════════════
Módulo: data_collector.py
Coleta de curvas de luz do telescópio Kepler via lightkurve
══════════════════════════════════════════════════════════════
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pipeline.utils.logger import logger
# ...
# ... (rest of StellarDataCollector)
# ...

class IndustrialDataCollector:
    """
    Responsável por baixar e cachear o dataset AI4I 2020 Predictive Maintenance.
    Fonte: UCI Machine Learning Repository
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.csv_path = os.path.join(self.cache_dir, "ai4i2020.csv")

    def download_dataset(self, url: str) -> pd.DataFrame:
        """
        Baixa o dataset se não estiver em cache.
        """
        if os.path.exists(self.csv_path):
            logger.info(f"Dataset industrial em cache: {self.csv_path}")
            return pd.read_csv(self.csv_path)

        logger.info(f"Baixando dataset industrial de: {url}...")
        try:
            df = pd.read_csv(url)
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Dataset salvo em: {self.csv_path}")
            return df
        except Exception as e:
            logger.error(f"Erro ao baixar dataset industrial: {e}")
            return pd.DataFrame()

    def load_local(self, path: str) -> pd.DataFrame:
        """Carrega um arquivo CSV local."""
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

warnings.filterwarnings("ignore")


class StellarDataCollector:
    """
    Responsável por baixar e cachear curvas de luz do Kepler.

    Utiliza a biblioteca lightkurve para acessar o arquivo MAST (Mikulski
    Archive for Space Telescopes) e obter dados fotométricos reais.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lk = None  # lazy import

    def _get_lightkurve(self):
        """Import lightkurve sob demanda (pode ser lento na primeira vez)."""
        if self._lk is None:
            try:
                import lightkurve as lk
                self._lk = lk
            except ImportError:
                raise ImportError(
                    "lightkurve não instalado. Execute: pip install lightkurve"
                )
        return self._lk

    def _cache_path(self, kic_id: str, quarter: int) -> str:
        """Caminho do arquivo de cache para uma estrela."""
        safe_name = kic_id.replace(" ", "_")
        return os.path.join(self.cache_dir, f"{safe_name}_q{quarter}.pkl")

    def download_lightcurve(
        self, kic_id: str, quarter: int = 16, author: str = "Kepler"
    ) -> Optional[Dict]:
        """
        Baixa a curva de luz de uma estrela do Kepler.

        Retorna:
            Dict com 'time' (array), 'flux' (array), 'kic_id', 'quarter'
            ou None se o download falhar.
        """
        # Verificar cache
        cache_file = self._cache_path(kic_id, quarter)
        if os.path.exists(cache_file):
            logger.info(f"Cache de luz encontrado: {kic_id}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Download via lightkurve
        try:
            lk = self._get_lightkurve()
            logger.info(f"Baixando: {kic_id} (quarter {quarter})...")

            search = lk.search_lightcurve(kic_id, author=author, quarter=quarter)
            if len(search) == 0:
                logger.warning(f"Nenhum resultado para {kic_id}")
                return None

            lc_raw = search.download()
            if lc_raw is None:
                logger.error(f"Download falhou para {kic_id}")
                return None

            result = {
                "kic_id": kic_id,
                "quarter": quarter,
                "time": lc_raw.time.value.copy(),
                "flux": lc_raw.flux.value.copy(),
                "flux_err": (
                    lc_raw.flux_err.value.copy()
                    if hasattr(lc_raw, "flux_err") and lc_raw.flux_err is not None
                    else None
                ),
            }

            # Salvar em cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            logger.info(f"Download OK ({kic_id}): {len(result['flux'])} pontos")
            return result

        except Exception as e:
            logger.error(f"Erro ao baixar {kic_id}: {e}")
            return None

    def download_batch(
        self, star_list: List[Dict], n_jobs: int = 4
    ) -> List[Dict]:
        """
        Baixa curvas de luz de uma lista de estrelas em paralelo.
        """
        from joblib import Parallel, delayed

        def download_single(star_info):
            data = self.download_lightcurve(
                kic_id=star_info["kic_id"],
                quarter=star_info.get("quarter", 16),
                author=star_info.get("author", "Kepler"),
            )
            if data is not None:
                data["nome"] = star_info.get("nome", star_info["kic_id"])
                data["categoria"] = star_info.get("categoria", "desconhecida")
                return data
            return None

        # Execução paralela (backend threading pois o gargalo é I/O de rede)
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(download_single)(s) for s in star_list
        )
        
        # Filtra falhas
        return [r for r in results if r is not None]

    @staticmethod
    def generate_simulated_star(
        n_points: int = 1500,
        duration_days: float = 90.0,
        noise_level: float = 0.0008,
        dips: Optional[List[Tuple[float, float, float]]] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Gera uma curva de luz simulada realista.

        Args:
            n_points: Número de pontos na curva.
            duration_days: Duração da observação em dias.
            noise_level: Nível de ruído fotônico (σ).
            dips: Lista de (centro, profundidade, largura) para quedas.
            seed: Semente para reprodutibilidade.

        Returns:
            Dict com 'time' e 'flux'.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        time = np.linspace(0, duration_days, n_points)
        flux = np.ones(n_points)

        # Ruído fotônico
        flux += rng.normal(0, noise_level, n_points)

        # Oscilação de fundo (atividade estelar)
        period = rng.uniform(8, 20)
        amplitude = rng.uniform(0.0001, 0.0005)
        flux += amplitude * np.sin(2 * np.pi * time / period)

        # Quedas de brilho
        if dips is None:
            dips = [
                (18, 0.22, 0.8),
                (38, 0.08, 1.2),
                (62, 0.05, 0.6),
                (77, 0.03, 0.4),
            ]

        for center, depth, width in dips:
            mask = np.abs(time - center) < width * 3
            if not np.any(mask):
                continue
            gaussian = depth * np.exp(
                -((time[mask] - center) ** 2) / (2 * width ** 2)
            )
            # Assimetria: queda abrupta, recuperação lenta
            asymmetry = 1 + 0.5 * (time[mask] - center) / (width + 0.01)
            asymmetry = np.clip(asymmetry, 0.5, 1.5)
            flux[mask] -= gaussian * asymmetry

        flux = np.clip(flux, 0.75, 1.01)

        return {"time": time, "flux": flux, "kic_id": "SIMULADA", "quarter": 0}
