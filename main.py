"""Script orquestrador principal do projeto AEDI - Estudo de Caso ANDI (Grupo 03).

Executa o fluxo completo a partir de um único comando (`python main.py`),
garantindo a criação da estrutura de pastas, inicialização do logger de auditoria
e validação determinística do gerador de números aleatórios (SEED = 202603).
"""

import sys
from typing import List
from pathlib import Path

from src.config import GROUP_NAME, SEED
from src.utils.logger import get_logger
from src.utils.rng import disable_legacy_global_rng, validate_rng_determinism
from src.utils.setup_project import create_project_directories


def print_header() -> None:
    """Imprime o cabeçalho padronizado de execução no terminal."""
    print("================================")
    print("AEDI - ANDI")
    print(f"{GROUP_NAME}")
    print(f"Seed: {SEED}")
    print("================================\n")


def main() -> int:
    """Executa a Fase 0 de Setup e Governança e prepara o pipeline principal.

    Returns:
        Código de saída do sistema (0 para sucesso, 1 para falha).
    """
    # 1. Carregar configurações e exibir cabeçalho com a seed
    print_header()

    # 2. Criar automaticamente as pastas exigidas pelo projeto
    created_dirs: List[Path] = create_project_directories()
    print("[OK] Estrutura criada")

    # 3. Iniciar o logger de auditoria
    logger = get_logger(name="main")
    logger.info("Iniciando execução do pipeline AEDI - ANDI (%s).", GROUP_NAME)
    logger.info("Diretórios verificados/criados: %d pastas.", len(created_dirs))
    print("[OK] Logger iniciado")

    # 4. Bloquear RNG legado global e testar determinismo do RNG oficial
    disable_legacy_global_rng()
    rng_ok: bool = validate_rng_determinism(seed=SEED)
    if rng_ok:
        logger.info("Validação determinística do RNG (PCG64) concluída com sucesso.")
        print("[OK] RNG validado\n")

    # 5. Encerrar execução da Fase 0
    logger.info("Execução da Fase 0 finalizada com sucesso.")
    print("Execução finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())