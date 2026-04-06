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
from typing import Dict, List, Optional, Tuple

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
            print(f"    [OK] Cache encontrado: {kic_id}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Download via lightkurve
        try:
            lk = self._get_lightkurve()
            print(f"    [v] Baixando: {kic_id} (quarter {quarter})...")

            search = lk.search_lightcurve(kic_id, author=author, quarter=quarter)
            if len(search) == 0:
                print(f"    [X] Nenhum resultado para {kic_id}")
                return None

            lc_raw = search.download()
            if lc_raw is None:
                print(f"    [X] Download falhou para {kic_id}")
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

            print(f"    [OK] {kic_id}: {len(result['flux'])} pontos")
            return result

        except Exception as e:
            print(f"    [X] Erro ao baixar {kic_id}: {e}")
            return None

    def download_batch(
        self, star_list: List[Dict]
    ) -> List[Dict]:
        """
        Baixa curvas de luz de uma lista de estrelas.

        Args:
            star_list: Lista de dicts com 'kic_id', 'quarter', etc.

        Returns:
            Lista de dicts com dados (exclui falhas).
        """
        results = []
        for star_info in star_list:
            data = self.download_lightcurve(
                kic_id=star_info["kic_id"],
                quarter=star_info.get("quarter", 16),
                author=star_info.get("author", "Kepler"),
            )
            if data is not None:
                data["nome"] = star_info.get("nome", star_info["kic_id"])
                data["categoria"] = star_info.get("categoria", "desconhecida")
                results.append(data)
        return results

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
