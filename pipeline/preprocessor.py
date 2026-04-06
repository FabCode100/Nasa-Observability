"""
══════════════════════════════════════════════════════════════
Módulo: preprocessor.py
Limpeza e normalização de curvas de luz
══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, Optional


class LightCurvePreprocessor:
    """
    Pipeline de pré-processamento para curvas de luz fotométricas.

    Etapas:
        1. Remoção de NaN/Inf
        2. Sigma-clipping (remoção de outliers extremos)
        3. Flatten (remoção de tendências de longo prazo)
        4. Normalização em torno de 1.0
    """

    def __init__(self, outlier_sigma: float = 5.0, flatten_window: int = 401):
        """
        Args:
            outlier_sigma: Número de sigmas para sigma-clipping.
            flatten_window: Tamanho da janela do filtro Savitzky-Golay (ímpar).
        """
        self.outlier_sigma = outlier_sigma
        self.flatten_window = flatten_window

    def remove_nans(self, time: np.ndarray, flux: np.ndarray):
        """Remove pontos com NaN ou Inf."""
        mask = np.isfinite(flux) & np.isfinite(time)
        return time[mask], flux[mask]

    def sigma_clip(self, time: np.ndarray, flux: np.ndarray):
        """Remove outliers por sigma-clipping."""
        median = np.median(flux)
        std = np.std(flux)
        mask = np.abs(flux - median) < self.outlier_sigma * std
        return time[mask], flux[mask]

    def flatten(self, flux: np.ndarray) -> np.ndarray:
        """
        Remove tendências de longo prazo usando filtro de média móvel.

        Usa uma implementação simples de convolução para evitar
        dependência do lightkurve nesta etapa.
        """
        window = min(self.flatten_window, len(flux))
        if window % 2 == 0:
            window += 1

        # Média móvel com padding nas bordas
        kernel = np.ones(window) / window
        trend = np.convolve(flux, kernel, mode="same")

        # Corrigir bordas (onde a convolução é parcial)
        half = window // 2
        for i in range(half):
            trend[i] = np.mean(flux[: i + half + 1])
            trend[-(i + 1)] = np.mean(flux[-(i + half + 1) :])

        # Dividir pelo trend para remover variações de longo prazo
        flattened = flux / trend
        return flattened

    def normalize(self, flux: np.ndarray) -> np.ndarray:
        """Normaliza o fluxo para que a mediana seja 1.0."""
        median = np.median(flux)
        if median == 0 or not np.isfinite(median):
            return flux
        return flux / median

    def process(self, star_data: Dict) -> Dict:
        """
        Aplica o pipeline completo de pré-processamento.

        Args:
            star_data: Dict com 'time' e 'flux' (arrays).

        Returns:
            Dict atualizado com 'time' e 'flux' processados,
            mais 'n_original' e 'n_final' para diagnóstico.
        """
        time = np.array(star_data["time"], dtype=float)
        flux = np.array(star_data["flux"], dtype=float)

        n_original = len(flux)

        # 1. Remove NaN/Inf
        time, flux = self.remove_nans(time, flux)

        # 2. Sigma-clipping
        time, flux = self.sigma_clip(time, flux)

        # 3. Flatten (remove tendências)
        if len(flux) > self.flatten_window:
            flux = self.flatten(flux)

        # 4. Normalização
        flux = self.normalize(flux)

        result = dict(star_data)
        result["time"] = time
        result["flux"] = flux
        result["n_original"] = n_original
        result["n_final"] = len(flux)

        return result

    def process_batch(self, star_list):
        """Processa uma lista de estrelas."""
        results = []
        for star_data in star_list:
            try:
                processed = self.process(star_data)
                if processed["n_final"] > 100:  # mínimo de pontos
                    results.append(processed)
                else:
                    kic = star_data.get("kic_id", "?")
                    print(f"    [!] {kic}: poucos pontos apos limpeza ({processed['n_final']})")
            except Exception as e:
                kic = star_data.get("kic_id", "?")
                print(f"    [X] Erro ao processar {kic}: {e}")
        return results
