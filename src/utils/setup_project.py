"""Utilitário para criação e verificação automática da árvore de diretórios do projeto."""

from pathlib import Path
from typing import List

from src.config import (
    DATA_DIR,
    DOCS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    TABLES_DIR,
)


def get_required_directories() -> List[Path]:
    """Retorna a lista de diretórios de dados, saídas, documentos e logs exigidos.

    Returns:
        Lista de objetos `Path` representando as pastas necessárias para o pipeline.
    """
    return [
        DATA_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        DOCS_DIR,
        LOGS_DIR,
    ]


def create_project_directories() -> List[Path]:
    """Cria automaticamente todos os diretórios de trabalho caso ainda não existam.

    Utiliza `mkdir(parents=True, exist_ok=True)` para operação idempotente,
    permitindo reexecuções seguras de `python main.py`.

    Returns:
        Lista com os caminhos dos diretórios verificados/criados.
    """
    directories: List[Path] = get_required_directories()
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories