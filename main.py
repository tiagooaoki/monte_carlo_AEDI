"""Script orquestrador principal do projeto AEDI - Estudo de Caso ANDI (Grupo 03).

Executa o fluxo completo a partir de um único comando (`python main.py`):
    1. Fase 0: Setup de diretórios, logger de auditoria e validação determinística do RNG.
    2. Fase 1 (Questão 1): Validação das transformações inversas F⁻¹(U) e geração por
       Simulação de Monte Carlo das 5 bases de dados oficiais (.csv).
    3. Fase 1 (Questão 2): Verificação estatística do gerador (lambda, sobredispersão,
       frequências de tipos e materialização) com exportação de tabelas e figuras.
"""

import sys
from pathlib import Path
from typing import List

from numpy.random import Generator

from src.config import GROUP_NAME, OUTPUT_DIR, SEED, TABLES_DIR
from src.parte_a.generator import GeneratedDatasets, run_parte_a_generator
from src.parte_a.inverse_transform import run_inverse_transform_validation
from src.parte_a.validator import run_parte_a_validation
from src.utils.logger import get_logger
from src.utils.rng import disable_legacy_global_rng, get_rng, validate_rng_determinism
from src.utils.setup_project import create_project_directories


def print_header() -> None:
    """Imprime o cabeçalho padronizado de execução no terminal."""
    print("================================")
    print("AEDI - ANDI")
    print(f"{GROUP_NAME}")
    print(f"Seed: {SEED}")
    print("================================\n")


def main() -> int:
    """Executa o pipeline do projeto (Setup -> Questão 1 -> Questão 2).

    Returns:
        Código de saída do sistema (0 para sucesso, 1 para falha).
    """
    # -------------------------------------------------------------------------
    # FASE 0: SETUP E GOVERNANÇA
    # -------------------------------------------------------------------------
    print_header()

    created_dirs: List[Path] = create_project_directories()
    validation_dir: Path = OUTPUT_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Estrutura criada")

    logger = get_logger(name="main")
    logger.info("Iniciando execução do pipeline AEDI - ANDI (%s).", GROUP_NAME)
    logger.info("Diretórios verificados/criados: %d pastas.", len(created_dirs) + 1)
    print("[OK] Logger iniciado")

    disable_legacy_global_rng()
    rng_ok: bool = validate_rng_determinism(seed=SEED)
    if rng_ok:
        logger.info("Validação determinística do RNG (PCG64) concluída com sucesso.")
        print("[OK] RNG validado")

    # -------------------------------------------------------------------------
    # FASE 1: QUESTÃO 1 — IMPLEMENTAÇÃO DO GERADOR E TRANSFORMADAS INVERSAS
    # -------------------------------------------------------------------------
    # Instancia o gerador principal da Parte A com a semente oficial (202603)
    rng_gerador: Generator = get_rng(SEED)
    datasets: GeneratedDatasets = run_parte_a_generator(rng=rng_gerador)
    print(
        f"[OK] Questão 1-a concluída: 5 bases geradas "
        f"({len(datasets.unidades)} unidades, {len(datasets.ataques)} ataques, "
        f"{len(datasets.perdas)} perdas)"
    )

    # Instancia gerador dedicado para o experimento de validação das transformadas inversas (Q1-b)
    rng_inversa: Generator = get_rng(SEED)
    run_inverse_transform_validation(
        rng=rng_inversa,
        validation_dir=validation_dir,
        tables_dir=TABLES_DIR,
    )
    print("[OK] Questão 1-b concluída: Transformadas inversas F⁻¹(U) validadas (KS test)")

    # -------------------------------------------------------------------------
    # FASE 1: QUESTÃO 2 — VERIFICAÇÃO ESTATÍSTICA DO GERADOR
    # -------------------------------------------------------------------------
    resultados_q2 = run_parte_a_validation(
        datasets=datasets,
        tables_dir=TABLES_DIR,
        validation_dir=validation_dir,
    )
    taxa_mat_obs: float = float(
        resultados_q2["materializacao"].loc[0, "taxa_materializacao_observada"]
    )
    media_p_teo: float = float(
        resultados_q2["materializacao"].loc[0, "media_p_i_programada"]
    )
    print(
        f"[OK] Questão 2 concluída: Tabelas e gráficos de validação exportados "
        f"(Materialização Obs={taxa_mat_obs:.4f} vs Teo={media_p_teo:.4f})\n"
    )

    logger.info("Execução da Fase 1 (Questões 1 e 2) finalizada com sucesso.")
    print("Execução finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())