"""Script orquestrador principal do projeto AEDI - Estudo de Caso ANDI (Grupo 03).

Executa o fluxo completo a partir de um único comando (`python main.py`):
    1. Fase 0: Setup de diretórios, logger de auditoria e validação determinística do RNG.
    2. Fase 1 (Questões 1 e 2): Geração por Simulação de Monte Carlo das 5 bases
       oficiais (.csv), validação das transformadas inversas F⁻¹(U) e verificação do gerador.
    3. Fase 2 (Questões 3, 4, 5 e 6): Estatística descritiva, quantis e regras de
       Sturges/Freedman-Diaconis (Q3), box-plots, matrizes de correlação e scatterplots (Q4),
       Teorema de Bayes e independência (Q5), e ajuste de distribuições com testes KS
       e gráficos Q-Q (Q6).
"""

import sys
from pathlib import Path
from typing import List

from numpy.random import Generator

from src.config import FIGURES_DIR, GROUP_NAME, OUTPUT_DIR, SEED, TABLES_DIR
from src.parte_a.generator import GeneratedDatasets, run_parte_a_generator
from src.parte_a.inverse_transform import run_inverse_transform_validation
from src.parte_a.validator import run_parte_a_validation
from src.parte_b.descriptive import analyze_q3_descriptive, analyze_q4_visualizations
from src.parte_b.probability_bayes import analyze_q5_bayes, analyze_q6_distributions
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
    """Executa o pipeline do projeto (Fase 0 -> Fase 1 -> Fase 2).

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
    # FASE 1: PARTE A — QUESTÕES 1 E 2 (GERADOR SMC E VERIFICAÇÃO)
    # -------------------------------------------------------------------------
    rng_gerador: Generator = get_rng(SEED)
    datasets: GeneratedDatasets = run_parte_a_generator(rng=rng_gerador)
    print(
        f"[OK] Questão 1-a concluída: 5 bases geradas "
        f"({len(datasets.unidades)} unidades, {len(datasets.ataques)} ataques, "
        f"{len(datasets.perdas)} perdas)"
    )

    rng_inversa: Generator = get_rng(SEED)
    run_inverse_transform_validation(
        rng=rng_inversa,
        validation_dir=validation_dir,
        tables_dir=TABLES_DIR,
    )
    print("[OK] Questão 1-b concluída: Transformadas inversas F⁻¹(U) validadas (KS test)")

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
        f"(Materialização Obs={taxa_mat_obs:.4f} vs Teo={media_p_teo:.4f})"
    )

    # -------------------------------------------------------------------------
    # FASE 2: PARTE B (BLOCO 1) — QUESTÕES 3, 4, 5 E 6
    # -------------------------------------------------------------------------
    df_q3 = analyze_q3_descriptive(
        df_perdas=datasets.perdas,
        df_ataques=datasets.ataques,
        tables_dir=TABLES_DIR,
        figures_dir=FIGURES_DIR,
    )
    mediana_perda: float = float(df_q3.loc[0, "p50"])
    p95_perda: float = float(df_q3.loc[0, "p95"])
    print(
        f"[OK] Questão 3 concluída: Descritiva e histogramas gerados "
        f"(P50=R$ {mediana_perda:,.2f}, P95=R$ {p95_perda:,.2f})"
    )

    analyze_q4_visualizations(
        datasets=datasets,
        tables_dir=TABLES_DIR,
        figures_dir=FIGURES_DIR,
    )
    print("[OK] Questão 4 concluída: Box-plots, Heatmap de correlação e Scatterplots gerados")

    res_q5 = analyze_q5_bayes(
        df_ataques=datasets.ataques,
        tables_dir=TABLES_DIR,
    )
    df_bayes_soc = res_q5["bayes_soc"].set_index("metrica")
    p_soc_dado_mat: float = float(
        df_bayes_soc.loc["P(SOC | Materializou) [Teorema de Bayes]", "valor"]
    )
    print(
        f"[OK] Questão 5 concluída: Tabelas de Bayes exportadas "
        f"(P(SOC | Materializou) = {p_soc_dado_mat:.4f})"
    )

    analyze_q6_distributions(
        datasets=datasets,
        tables_dir=TABLES_DIR,
        figures_dir=FIGURES_DIR,
    )
    print("[OK] Questão 6 concluída: Ajustes Exponencial/Lognormal, testes KS e Q-Q plots gerados\n")

    logger.info("Execução até a Fase 2 (Questões 1 a 6) finalizada com sucesso.")
    print("Execução finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())