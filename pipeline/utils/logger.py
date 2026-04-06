import os
import sys
from loguru import logger

# ══════════════════════════════════════════════════════════
# pipeline/utils/logger.py
# ⚖️ Projeto: TCC Análise de Anomalias (Astro/Industrial)
# Framework: Loguru (Modern Logging)
# ══════════════════════════════════════════════════════════

def setup_logger():
    """
    Configura o Loguru para saída colorida no Console e persistência em Arquivo.
    Inclui rotação e retenção de logs.
    """
    # Limpar configurações padrão
    logger.remove()

    # Formato personalizado
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Handler para Console (Colorido)
    logger.add(
        sys.stdout, 
        format=log_format, 
        level="INFO", 
        colorize=True
    )

    # Handler para Arquivo (Log persistente)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline_{time:YYYY-MM-DD}.log")

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",    # Rotaciona quando atingir 10MB
        retention="10 days", # Mantém logs por 10 dias
        compression="zip",   # Compacta logs antigos
        encoding="utf-8"
    )

    return logger

# Instância global configurada
# O loguru já exporta um objeto 'logger', nós apenas o configuramos.
setup_logger()

# Exportamos para manter compatibilidade com o restante do projeto
# usage: from pipeline.utils.logger import logger
__all__ = ["logger"]
