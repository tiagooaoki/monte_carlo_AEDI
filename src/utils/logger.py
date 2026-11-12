"""Módulo de configuração de logging estruturado e auditoria de execução.

Registra todos os eventos de execução na pasta `logs/` com carimbo de tempo,
identificação do módulo e semente oficial utilizada pelo Grupo 03.
"""

import logging
from pathlib import Path
from typing import Optional

from src.config import LOGS_DIR, SEED


class SeedContextFilter(logging.Filter):
    """Filtro que injeta a constante `SEED` em cada registro de log."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self.seed: int = seed

    def filter(self, record: logging.LogRecord) -> bool:
        """Adiciona o atributo `seed` ao `LogRecord` antes da formatação."""
        record.seed = self.seed
        return True


def get_logger(
    name: str = "AEDI_ANDI",
    log_filename: str = "pipeline_grupo_03.log",
    seed: int = SEED,
    console_output: bool = False,
) -> logging.Logger:
    """Configura e retorna um logger padronizado para auditoria do projeto.

    Args:
        name: Nome do logger (normalmente o nome do módulo `__name__`).
        log_filename: Nome do arquivo de log dentro de `logs/`.
        seed: Semente ativa a ser estampada em todas as linhas do log.
        console_output: Se `True`, também envia mensagens de log para o terminal.

    Returns:
        Instância configurada de `logging.Logger`.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path: Path = LOGS_DIR / log_filename

    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Evita duplicação de handlers em chamadas repetidas a get_logger()
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | Seed=%(seed)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        seed_filter = SeedContextFilter(seed=seed)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(seed_filter)
        logger.addHandler(file_handler)

        if console_output:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            stream_handler.addFilter(seed_filter)
            logger.addHandler(stream_handler)

    return logger