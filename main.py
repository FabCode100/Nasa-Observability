"""
══════════════════════════════════════════════════════════════
TCC Astrofísica — Pipeline Principal
Detecção de Anomalias Estelares em Dados do Telescópio Kepler

Execução:
    python main.py

Após a execução, inicie o dashboard com:
    streamlit run dashboard.py
══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pickle
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline.data_collector import StellarDataCollector
from pipeline.preprocessor import LightCurvePreprocessor
from pipeline.feature_engineer import (
    PhotometricFeatureExtractor,
    generate_simulated_features,
    FEATURE_NAMES,
)
from pipeline.models import AnomalyDetector


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_step(step: int, total: int, text: str):
    print(f"\n[{step}/{total}] {text}")


def run_pipeline():
    """Executa o pipeline completo de detecção de anomalias."""

    print_header("PIPELINE DE DETECCAO DE ANOMALIAS ESTELARES")
    print("  Telescopio Kepler / NASA")
    print("  Isolation Forest + K-Means + PCA")
    print(f"{'=' * 60}")

    # ── PASSO 1: Coleta de dados ─────────────────────────────
    print_step(1, 5, "Coletando curvas de luz do Kepler...")


    collector = StellarDataCollector(cache_dir=config.CACHE_DIR)

    # Baixar estrela alvo
    print("\n  -- Estrela Alvo --")
    target_data = collector.download_lightcurve(
        kic_id=config.TARGET_STAR["kic_id"],
        quarter=config.TARGET_STAR["quarter"],
        author=config.TARGET_STAR["author"],
    )

    # Fallback para dados simulados
    if target_data is None:
        print("  [!] Usando dados simulados para a estrela alvo")
        target_data = collector.generate_simulated_star(seed=42)
        target_data["kic_id"] = config.TARGET_STAR["kic_id"]

    target_data["nome"] = config.TARGET_STAR["nome"]
    target_data["categoria"] = "ALVO"

    # Baixar estrelas de referência
    reference_stars = []
    for categoria, stars in config.REFERENCE_STARS.items():
        print(f"\n  -- Referencia: {categoria} --")
        for star_info in stars:
            star_info_copy = dict(star_info)
            star_info_copy["categoria"] = categoria
            data = collector.download_lightcurve(
                kic_id=star_info["kic_id"],
                quarter=star_info.get("quarter", 16),
            )
            if data is not None:
                data["nome"] = star_info.get("nome", star_info["kic_id"])
                data["categoria"] = categoria
                reference_stars.append(data)

    print(f"\n    -> Total de estrelas reais coletadas: {1 + len(reference_stars)}")

    # ── PASSO 2: Pré-processamento ───────────────────────────
    print_step(2, 5, "Pre-processando curvas de luz...")

    preprocessor = LightCurvePreprocessor(
        outlier_sigma=config.PREPROCESS["outlier_sigma"],
        flatten_window=config.PREPROCESS["flatten_window"],
    )

    # Processar alvo
    target_processed = preprocessor.process(target_data)
    print(f"    -> Alvo: {target_processed['n_original']} -> {target_processed['n_final']} pontos")

    # Processar referências
    reference_processed = preprocessor.process_batch(reference_stars)
    for star in reference_processed:
        print(f"    -> {star['kic_id']}: {star['n_original']} -> {star['n_final']} pontos")

    # ── PASSO 3: Extração de features ────────────────────────
    print_step(3, 5, "Extraindo features fotometricas...")

    extractor = PhotometricFeatureExtractor()

    # Features da estrela alvo
    target_enriched = extractor.extract_from_star(target_processed)
    print(f"\n    Features de {target_enriched['kic_id']}:")
    for k, v in target_enriched["features"].items():
        print(f"      {k:12s} = {v:.6f}")

    # Features das estrelas de referência
    reference_enriched = extractor.extract_batch(reference_processed)

    # Gerar população simulada para complementar
    print(f"\n    Gerando populacao simulada complementar...")
    sim_cfg = config.SIMULATED_POP
    sim_ranges = sim_cfg["ranges"]

    sim_estaveis = generate_simulated_features(
        sim_cfg["n_estaveis"],
        sim_ranges["estaveis"]["std"],
        sim_ranges["estaveis"]["range"],
        sim_ranges["estaveis"]["skew"],
        seed=sim_cfg["seed"],
    )
    sim_variaveis = generate_simulated_features(
        sim_cfg["n_variaveis"],
        sim_ranges["variaveis"]["std"],
        sim_ranges["variaveis"]["range"],
        sim_ranges["variaveis"]["skew"],
        seed=sim_cfg["seed"] + 1,
    )
    sim_extremas = generate_simulated_features(
        sim_cfg["n_extremas"],
        sim_ranges["extremas"]["std"],
        sim_ranges["extremas"]["range"],
        sim_ranges["extremas"]["skew"],
        seed=sim_cfg["seed"] + 2,
    )

    # Construir matriz de features completa
    # Ordem: referências reais → simuladas → alvo (último)
    real_matrix = extractor.build_feature_matrix(reference_enriched)
    target_vector = extractor.features_to_vector(target_enriched["features"])

    X_full = np.vstack([
        real_matrix,
        sim_estaveis,
        sim_variaveis,
        sim_extremas,
        target_vector.reshape(1, -1),
    ])

    # Labels de tipo para cada entrada
    labels = (
        [star.get("categoria", "?") for star in reference_enriched]
        + ["estável_sim"] * len(sim_estaveis)
        + ["variável_sim"] * len(sim_variaveis)
        + ["extrema_sim"] * len(sim_extremas)
        + ["ALVO"]
    )

    # Nomes para cada entrada
    names = (
        [star.get("kic_id", "?") for star in reference_enriched]
        + [f"SIM_EST_{i}" for i in range(len(sim_estaveis))]
        + [f"SIM_VAR_{i}" for i in range(len(sim_variaveis))]
        + [f"SIM_EXT_{i}" for i in range(len(sim_extremas))]
        + [config.TARGET_STAR["kic_id"]]
    )

    print(f"    -> {len(X_full)} estrelas no dataset total "
          f"({len(real_matrix)} reais + "
          f"{len(sim_estaveis)+len(sim_variaveis)+len(sim_extremas)} simuladas + 1 alvo)")

    # ── PASSO 4: Modelos de detecção ─────────────────────────
    print_step(4, 5, "Rodando Isolation Forest + K-Means + PCA...")

    detector = AnomalyDetector(
        if_params=config.ISOLATION_FOREST,
        km_params=config.KMEANS,
        pca_params=config.PCA_PARAMS,
    )
    detector.fit(X_full)

    summary = detector.get_summary(target_index=-1)
    target_result = summary["target"]

    print(f"    -> Isolation Forest: "
          f"{'ANOMALIA' if target_result['is_anomaly'] else 'NORMAL'} "
          f"(score={target_result['if_score']:.4f})")
    print(f"    -> K-Means cluster: C{target_result['cluster']}")
    print(f"    -> PCA variancia: "
          f"{summary['pca_variance'][0]:.1%} + {summary['pca_variance'][1]:.1%} = "
          f"{sum(summary['pca_variance']):.1%}")
    print(f"    -> Silhouette Score: {summary['silhouette_score']:.3f}")
    print(f"    -> Total de anomalias detectadas: {summary['n_anomalies']}/{summary['n_total']}")

    # ── PASSO 5: Persistir resultados ────────────────────────
    print_step(5, 5, "Salvando resultados para o dashboard...")


    results = {
        "target_star": target_enriched,
        "reference_stars": reference_enriched,
        "target_processed": {
            "time": target_processed["time"].tolist(),
            "flux": target_processed["flux"].tolist(),
            "kic_id": target_processed["kic_id"],
            "nome": target_processed.get("nome", ""),
        },
        "reference_processed": [
            {
                "time": s["time"].tolist(),
                "flux": s["flux"].tolist(),
                "kic_id": s["kic_id"],
                "nome": s.get("nome", ""),
                "categoria": s.get("categoria", ""),
                "features": s.get("features", {}),
            }
            for s in reference_enriched
        ],
        "feature_matrix": X_full.tolist(),
        "labels": labels,
        "names": names,
        "feature_names": FEATURE_NAMES,
        "model_summary": summary,
        "if_scores": detector.if_scores.tolist(),
        "if_predictions": detector.if_predictions.tolist(),
        "cluster_labels": detector.cluster_labels.tolist(),
        "pca_coords": detector.X_pca.tolist(),
        "pca_variance": detector.pca_variance.tolist(),
    }

    results_path = os.path.join(config.RESULTS_DIR, "pipeline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"    -> Resultados salvos em: {results_path}")

    # -- Relatorio final --
    print_header("RELATORIO DE ANOMALIA - " + config.TARGET_STAR["kic_id"])

    tc = target_result["cluster"]
    hyp = config.CLUSTER_HYPOTHESES.get(tc, {})

    print(f"  Isolation Forest:  {'ANOMALIA [!]' if target_result['is_anomaly'] else 'NORMAL [OK]'}"
          f" (score={target_result['if_score']:.4f})")
    print(f"  Cluster K-Means:   C{tc} - {hyp.get('nome', '?')}")
    print(f"  Silhouette Score:  {summary['silhouette_score']:.3f}")
    print(f"\n  Features extraidas:")
    for k, v in target_enriched["features"].items():
        print(f"    {k:12s} = {v:.6f}")
    print(f"\n  Hipotese principal:")
    for linha in hyp.get("hipotese", "N/A").split("\n"):
        print(f"    {linha}")

    print(f"\n{'=' * 60}")
    print("  Dashboard: execute  streamlit run dashboard.py")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    run_pipeline()
