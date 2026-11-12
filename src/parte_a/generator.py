"""Módulo gerador por Simulação de Monte Carlo da Parte A (Questão 1-a).

Implementa as cinco etapas sequenciais do modelo gerador da ANDI (Etapas A1 a A5)
e constrói as duas bases derivadas (`painel_trimestral.csv` e `serie_mensal.csv`),
documentando explicitamente os quatro passos metodológicos da Seção 12.2:
    - PASSO 1 (MODELAR): Especificação das equações estruturais e parâmetros.
    - PASSO 2 (SORTEAR): Geração das variáveis estocásticas via RNG e F⁻¹(U).
    - PASSO 3 (CALCULAR): Avaliação das funções de frequência, logit e severidade.
    - PASSO 4 (ANALISAR / AGREGAR): Consolidação relacional, painel e série mensal.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

from src.config import (
    ANO_INICIAL,
    ATAQUES_CSV,
    ETAPA_A1,
    ETAPA_A2,
    ETAPA_A3,
    ETAPA_A4,
    ETAPA_A5,
    FATORES_SAZONAIS,
    N_MESES,
    N_TRIMESTRES,
    N_UNIDADES,
    PAINEL_TRIMESTRAL_CSV,
    PERDAS_CSV,
    SERIE_MENSAL_CSV,
    TIPOS_ATAQUE,
    UNIDADES_CSV,
)
from src.parte_a.inverse_transform import inverse_exponential, inverse_triangular
from src.utils.logger import get_logger

logger = get_logger(name="parte_a.generator")

# Número de dias de cada mês do calendário conforme o enunciado (28, 30 ou 31)
DIAS_POR_MES: Dict[int, int] = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


@dataclass(frozen=True)
class GeneratedDatasets:
    """Contêiner imutável com as 5 bases oficiais e a grade teórica de taxas lambda."""

    unidades: pd.DataFrame
    ataques: pd.DataFrame
    perdas: pd.DataFrame
    painel_trimestral: pd.DataFrame
    serie_mensal: pd.DataFrame
    grade_lambda_mensal: pd.DataFrame


def generate_etapa_a1_unidades(rng: Generator) -> pd.DataFrame:
    """Executa a Etapa A1: caracteriza o perfil fixo das 24 unidades de negócio.

    Como cada distribuição é sorteada a partir de um gerador uniforme U ~ U(0, 1):
        - servicos: Uniforme discreta em {10, ..., 30} via floor(a + (b - a + 1)*U).
        - patch: Triangular(60, 80, 95) via transformação inversa F⁻¹(U) em dois ramos.
        - treino: Uniforme contínua(0, 12) via transformação linear a + (b - a)*U.
        - soc: Bernoulli(0,40) via função indicadora I(U < p).

    Args:
        rng: Instância isolada de `numpy.random.Generator`.

    Returns:
        DataFrame `unidades` com 24 linhas e colunas `unidade, servicos, patch, treino, soc`.
    """
    # PASSO 1 (MODELAR): Definição do suporte e parâmetros de perfil via ETAPA_A1
    unidades_id: np.ndarray = np.arange(1, N_UNIDADES + 1, dtype=int)

    # PASSO 2 (SORTEAR) & PASSO 3 (CALCULAR): Sorteio via transformações de U ~ U(0,1)
    u_servicos: np.ndarray = rng.uniform(0.0, 1.0, size=N_UNIDADES)
    amplitude_servicos: int = ETAPA_A1.servicos_max - ETAPA_A1.servicos_min + 1
    servicos: np.ndarray = (
        ETAPA_A1.servicos_min + np.floor(amplitude_servicos * u_servicos)
    ).astype(int)

    patch: np.ndarray = inverse_triangular(
        rng=rng,
        left=ETAPA_A1.patch_min,
        mode=ETAPA_A1.patch_mode,
        right=ETAPA_A1.patch_max,
        size=N_UNIDADES,
    )

    u_treino: np.ndarray = rng.uniform(0.0, 1.0, size=N_UNIDADES)
    treino: np.ndarray = ETAPA_A1.treino_min + (
        ETAPA_A1.treino_max - ETAPA_A1.treino_min
    ) * u_treino

    u_soc: np.ndarray = rng.uniform(0.0, 1.0, size=N_UNIDADES)
    soc: np.ndarray = (u_soc < ETAPA_A1.soc_prob).astype(int)

    df_unidades = pd.DataFrame(
        {
            "unidade": unidades_id,
            "servicos": servicos,
            "patch": np.round(patch, 4),
            "treino": np.round(treino, 4),
            "soc": soc,
        }
    )
    logger.info("Etapa A1 concluída: %d unidades geradas.", len(df_unidades))
    return df_unidades


def generate_etapa_a2_contagens(
    rng: Generator,
    df_unidades: pd.DataFrame,
) -> pd.DataFrame:
    """Executa a Etapa A2: calcula taxas lambda e sorteia N_{u,t} ~ Poisson(lambda_{u,t}).

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        df_unidades: DataFrame gerado na Etapa A1 com o perfil das 24 unidades.

    Returns:
        DataFrame `grade_lambda_mensal` com 864 linhas (24 unidades x 36 meses)
        contendo `lambda_trim`, `lambda_mensal` e a contagem sorteada `n_ataques`.
    """
    registros = []

    # PASSO 1 (MODELAR): Horizonte t = 1..36, q(t) = ceil(t/3), m(t) = 1..12
    for t in range(1, N_MESES + 1):
        trimestre_q: int = int(np.ceil(t / 3.0))
        mes_m: int = ((t - 1) % 12) + 1
        ano: int = ANO_INICIAL + ((t - 1) // 12)
        mes_ref: str = f"{ano:04d}-{mes_m:02d}"

        s_m: float = FATORES_SAZONAIS[mes_m]
        c_t: float = (
            ETAPA_A2.fator_campanha_c28
            if t == ETAPA_A2.mes_campanha_t
            else ETAPA_A2.fator_campanha_padrao
        )

        for row in df_unidades.itertuples(index=False):
            # PASSO 3 (CALCULAR): Taxa trimestral e taxa mensal esperada
            lambda_trim: float = (
                ETAPA_A2.beta_0
                + ETAPA_A2.beta_servicos * float(row.servicos)
                + ETAPA_A2.beta_patch * float(row.patch)
                + ETAPA_A2.beta_treino * float(row.treino)
                + ETAPA_A2.beta_soc * float(row.soc)
                + ETAPA_A2.beta_trimestre * float(trimestre_q)
            )
            lambda_mensal: float = (lambda_trim / 3.0) * s_m * c_t

            registros.append(
                {
                    "t": t,
                    "ano": ano,
                    "mes": mes_m,
                    "trimestre": trimestre_q,
                    "mes_ref": mes_ref,
                    "unidade": int(row.unidade),
                    "servicos": int(row.servicos),
                    "patch": float(row.patch),
                    "treino": float(row.treino),
                    "soc": int(row.soc),
                    "s_m": s_m,
                    "c_t": c_t,
                    "lambda_trim": lambda_trim,
                    "lambda_mensal": lambda_mensal,
                }
            )

    df_grade = pd.DataFrame(registros)

    # PASSO 2 (SORTEAR): Contagem mensal N_{u,t} ~ Poisson(lambda_{u,t})
    df_grade["n_ataques"] = rng.poisson(lam=df_grade["lambda_mensal"].to_numpy())

    logger.info(
        "Etapa A2 concluída: %d células (unidade x mês), total de %d ataques sorteados.",
        len(df_grade),
        int(df_grade["n_ataques"].sum()),
    )
    return df_grade


def generate_etapas_a3_a4_ataques(
    rng: Generator,
    df_grade: pd.DataFrame,
) -> pd.DataFrame:
    """Executa as Etapas A3 e A4: sorteia atributos de cada ataque e sua materialização.

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        df_grade: DataFrame da Etapa A2 com as contagens `n_ataques` por unidade e mês.

    Returns:
        DataFrame `ataques` contendo todos os incidentes e variáveis de auditoria.
    """
    # Expande a grade proporcionalmente a n_ataques de cada célula (u, t)
    df_ativos = df_grade[df_grade["n_ataques"] > 0].copy()
    indices_repetidos = np.repeat(df_ativos.index.to_numpy(), df_ativos["n_ataques"].to_numpy())
    df_exp = df_ativos.loc[indices_repetidos].reset_index(drop=True)

    total_ataques: int = len(df_exp)
    id_ataque: np.ndarray = np.arange(1, total_ataques + 1, dtype=int)

    # =========================================================================
    # ETAPA A3 — ATRIBUTOS DE CADA ATAQUE (PASSO 2: SORTEAR & PASSO 3: CALCULAR)
    # =========================================================================
    # 1. Dia do mês: Uniforme discreta entre 1 e o número de dias do mês (28, 30 ou 31)
    max_dias: np.ndarray = df_exp["mes"].map(DIAS_POR_MES).to_numpy(dtype=int)
    u_dia: np.ndarray = rng.uniform(0.0, 1.0, size=total_ataques)
    dias: np.ndarray = (1 + np.floor(max_dias * u_dia)).astype(int)

    datas = [
        f"{int(ano):04d}-{int(mes):02d}-{int(dia):02d}"
        for ano, mes, dia in zip(df_exp["ano"], df_exp["mes"], dias)
    ]

    # 2. Tipo de ataque: Multinomial(1, p) via inversão da acumulada com U ~ U(0,1)
    nomes_tipos = np.array([spec.nome for spec in TIPOS_ATAQUE], dtype=object)
    probs_tipos = np.array([spec.probabilidade for spec in TIPOS_ATAQUE], dtype=float)
    cum_probs = np.cumsum(probs_tipos)
    cum_probs[-1] = 1.0  # Garante fechamento exato em 1.0

    u_tipo: np.ndarray = rng.uniform(0.0, 1.0, size=total_ataques)
    idx_tipo: np.ndarray = np.searchsorted(cum_probs, u_tipo, side="right")
    tipos_sorteados: np.ndarray = nomes_tipos[idx_tipo]

    # 3. Fora do expediente: Bernoulli(p = 0,35)
    u_fora: np.ndarray = rng.uniform(0.0, 1.0, size=total_ataques)
    fora_expediente: np.ndarray = (u_fora < ETAPA_A3.fora_expediente_prob).astype(int)

    # 4. CVSS: Triangular por tipo de ataque via inverse_triangular, arredondado a 1 casa
    cvss_raw: np.ndarray = np.zeros(total_ataques, dtype=float)
    for idx_spec, spec in enumerate(TIPOS_ATAQUE):
        mask = idx_tipo == idx_spec
        n_mask: int = int(np.sum(mask))
        if n_mask > 0:
            cvss_raw[mask] = inverse_triangular(
                rng=rng,
                left=spec.cvss_min,
                mode=spec.cvss_mode,
                right=spec.cvss_max,
                size=n_mask,
            )
    cvss: np.ndarray = np.round(cvss_raw, ETAPA_A3.cvss_decimais)

    # 5. Tempo de detecção (h): Exponencial via inverse_exponential (4h c/ SOC, 12h s/ SOC)
    # multiplicado por 1,5 quando fora_expediente == 1
    soc_vetor: np.ndarray = df_exp["soc"].to_numpy(dtype=int)
    tempo_base: np.ndarray = np.zeros(total_ataques, dtype=float)

    mask_com_soc = soc_vetor == 1
    n_com_soc: int = int(np.sum(mask_com_soc))
    if n_com_soc > 0:
        tempo_base[mask_com_soc] = inverse_exponential(
            rng=rng, theta=ETAPA_A3.theta_com_soc_h, size=n_com_soc
        )

    mask_sem_soc = ~mask_com_soc
    n_sem_soc: int = int(np.sum(mask_sem_soc))
    if n_sem_soc > 0:
        tempo_base[mask_sem_soc] = inverse_exponential(
            rng=rng, theta=ETAPA_A3.theta_sem_soc_h, size=n_sem_soc
        )

    fator_horario: np.ndarray = np.where(
        fora_expediente == 1, ETAPA_A3.multiplicador_fora_expediente, 1.0
    )
    tempo_deteccao_h: np.ndarray = tempo_base * fator_horario

    # =========================================================================
    # ETAPA A4 — MATERIALIZAÇÃO EM PERDA (MODELO LOGÍSTICO)
    # =========================================================================
    treino_vetor: np.ndarray = df_exp["treino"].to_numpy(dtype=float)
    logit_p: np.ndarray = (
        ETAPA_A4.beta_0
        + ETAPA_A4.beta_cvss * cvss
        + ETAPA_A4.beta_tempo * tempo_deteccao_h
        + ETAPA_A4.beta_soc * soc_vetor
        + ETAPA_A4.beta_treino * treino_vetor
        + ETAPA_A4.beta_fora_expediente * fora_expediente
    )
    p_materializacao: np.ndarray = 1.0 / (1.0 + np.exp(-logit_p))

    u_mat: np.ndarray = rng.uniform(0.0, 1.0, size=total_ataques)
    materializou: np.ndarray = (u_mat < p_materializacao).astype(int)

    df_ataques = pd.DataFrame(
        {
            "id_ataque": id_ataque,
            "unidade": df_exp["unidade"].to_numpy(dtype=int),
            "data": datas,
            "ano": df_exp["ano"].to_numpy(dtype=int),
            "mes": df_exp["mes"].to_numpy(dtype=int),
            "trimestre": df_exp["trimestre"].to_numpy(dtype=int),
            "dia": dias,
            "servicos": df_exp["servicos"].to_numpy(dtype=int),
            "patch": df_exp["patch"].to_numpy(dtype=float),
            "treino": df_exp["treino"].to_numpy(dtype=float),
            "soc": soc_vetor,
            "tipo": tipos_sorteados,
            "fora_expediente": fora_expediente,
            "cvss": cvss,
            "tempo_deteccao_h": np.round(tempo_deteccao_h, 4),
            "p_materializacao": np.round(p_materializacao, 6),
            "materializou": materializou,
        }
    )

    logger.info(
        "Etapas A3 e A4 concluídas: %d ataques processados, %d materializados (%.2f%%).",
        len(df_ataques),
        int(df_ataques["materializou"].sum()),
        100.0 * float(df_ataques["materializou"].mean()),
    )
    return df_ataques


def generate_etapa_a5_perdas(
    rng: Generator,
    df_ataques: pd.DataFrame,
) -> pd.DataFrame:
    """Executa a Etapa A5: sorteia a severidade Lognormal para ataques materializados.

    Equação estrutural:
        ln(perda_i) = ln(mediana_tipo) + 0.02 * tempo_deteccao_h + erro_i,
        onde erro_i ~ Normal(0, 0.9^2).

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        df_ataques: DataFrame gerado nas Etapas A3 e A4.

    Returns:
        DataFrame `perdas` contendo apenas os incidentes com `materializou == 1`.
    """
    df_mat = df_ataques[df_ataques["materializou"] == 1].copy().reset_index(drop=True)
    n_perdas: int = len(df_mat)

    mapa_mediana: Dict[str, float] = {
        spec.nome: spec.mediana_perda_rs for spec in TIPOS_ATAQUE
    }
    mediana_tipo: np.ndarray = df_mat["tipo"].map(mapa_mediana).to_numpy(dtype=float)

    # PASSO 2 (SORTEAR): Erro normal via Box-Muller (transformação a partir de Uniformes)
    # ou diretamente pelo gerador PCG64 encapsulado
    erro_normal: np.ndarray = rng.normal(
        loc=0.0, scale=ETAPA_A5.sigma_erro, size=n_perdas
    )

    # PASSO 3 (CALCULAR): Severidade na escala logarítmica e exponenciação para Reais (R$)
    tempo_h: np.ndarray = df_mat["tempo_deteccao_h"].to_numpy(dtype=float)
    ln_perda: np.ndarray = (
        np.log(mediana_tipo) + ETAPA_A5.beta_tempo * tempo_h + erro_normal
    )
    perda_rs: np.ndarray = np.exp(ln_perda)

    df_perdas = pd.DataFrame(
        {
            "id_perda": np.arange(1, n_perdas + 1, dtype=int),
            "id_ataque": df_mat["id_ataque"].to_numpy(dtype=int),
            "unidade": df_mat["unidade"].to_numpy(dtype=int),
            "data": df_mat["data"].to_numpy(),
            "tipo": df_mat["tipo"].to_numpy(),
            "tempo_deteccao_h": df_mat["tempo_deteccao_h"].to_numpy(dtype=float),
            "perda_rs": np.round(perda_rs, 2),
        }
    )

    logger.info(
        "Etapa A5 concluída: %d perdas geradas, perda total acumulada = R$ %.2f.",
        len(df_perdas),
        float(df_perdas["perda_rs"].sum()),
    )
    return df_perdas


def build_derived_datasets(
    df_unidades: pd.DataFrame,
    df_ataques: pd.DataFrame,
    df_perdas: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constrói via código as bases agregadas `painel_trimestral` (288) e `serie_mensal` (36).

    PASSO 4 (ANALISAR / AGREGAR - Seção 12.2): Consolida as três bases primárias sem
    intervenção manual, garantindo a presença de células com contagem zero.

    Args:
        df_unidades: Base `unidades.csv` (24 linhas).
        df_ataques: Base `ataques.csv`.
        df_perdas: Base `perdas.csv`.

    Returns:
        Tupla `(df_painel_trimestral, df_serie_mensal)`.
    """
    # Associa perda_rs de volta aos ataques via id_ataque
    df_atq_perda = df_ataques.merge(
        df_perdas[["id_ataque", "perda_rs"]],
        on="id_ataque",
        how="left",
    )
    df_atq_perda["perda_rs"] = df_atq_perda["perda_rs"].fillna(0.0)

    # -------------------------------------------------------------------------
    # 1. Base Derivada: painel_trimestral.csv (24 unidades x 12 trimestres = 288 linhas)
    # -------------------------------------------------------------------------
    malha_painel = pd.MultiIndex.from_product(
        [df_unidades["unidade"].to_numpy(), np.arange(1, N_TRIMESTRES + 1, dtype=int)],
        names=["unidade", "trimestre"],
    ).to_frame().reset_index(drop=True)

    agregado_trim = (
        df_atq_perda.groupby(["unidade", "trimestre"], as_index=False)
        .agg(
            n_ataques=("id_ataque", "count"),
            n_perdas=("materializou", "sum"),
            perda_total=("perda_rs", "sum"),
        )
    )

    df_painel = (
        malha_painel.merge(agregado_trim, on=["unidade", "trimestre"], how="left")
        .merge(df_unidades, on="unidade", how="left")
    )
    df_painel["n_ataques"] = df_painel["n_ataques"].fillna(0).astype(int)
    df_painel["n_perdas"] = df_painel["n_perdas"].fillna(0).astype(int)
    df_painel["perda_total"] = np.round(df_painel["perda_total"].fillna(0.0), 2)

    colunas_painel = [
        "unidade",
        "trimestre",
        "n_ataques",
        "n_perdas",
        "perda_total",
        "servicos",
        "patch",
        "treino",
        "soc",
    ]
    df_painel = df_painel[colunas_painel].sort_values(["unidade", "trimestre"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 2. Base Derivada: serie_mensal.csv (36 meses)
    # -------------------------------------------------------------------------
    meses_calendario = []
    for t in range(1, N_MESES + 1):
        ano_t: int = ANO_INICIAL + ((t - 1) // 12)
        mes_t: int = ((t - 1) % 12) + 1
        meses_calendario.append(
            {"ano": ano_t, "mes": mes_t, "mes_ref": f"{ano_t:04d}-{mes_t:02d}"}
        )
    df_malha_mensal = pd.DataFrame(meses_calendario)

    agregado_mensal = (
        df_atq_perda.groupby(["ano", "mes"], as_index=False)
        .agg(
            n_ataques=("id_ataque", "count"),
            n_perdas=("materializou", "sum"),
            perda_total=("perda_rs", "sum"),
        )
    )

    df_serie = df_malha_mensal.merge(agregado_mensal, on=["ano", "mes"], how="left")
    df_serie["n_ataques"] = df_serie["n_ataques"].fillna(0).astype(int)
    df_serie["n_perdas"] = df_serie["n_perdas"].fillna(0).astype(int)
    df_serie["perda_total"] = np.round(df_serie["perda_total"].fillna(0.0), 2)

    df_serie = df_serie[["mes_ref", "n_ataques", "n_perdas", "perda_total"]]

    logger.info(
        "Bases derivadas construídas: painel_trimestral (%d linhas) e serie_mensal (%d linhas).",
        len(df_painel),
        len(df_serie),
    )
    return df_painel, df_serie


def run_parte_a_generator(rng: Generator) -> GeneratedDatasets:
    """Executa integralmente o gerador da Parte A e salva os 5 arquivos CSV oficiais.

    Args:
        rng: Instância isolada de `numpy.random.Generator` inicializada com `SEED`.

    Returns:
        Objeto `GeneratedDatasets` contendo os DataFrames gerados em memória.
    """
    logger.info("Iniciando geração das bases de dados da Parte A (36 meses)...")

    df_unidades = generate_etapa_a1_unidades(rng=rng)
    df_grade = generate_etapa_a2_contagens(rng=rng, df_unidades=df_unidades)
    df_ataques = generate_etapas_a3_a4_ataques(rng=rng, df_grade=df_grade)
    df_perdas = generate_etapa_a5_perdas(rng=rng, df_ataques=df_ataques)
    df_painel, df_serie = build_derived_datasets(
        df_unidades=df_unidades,
        df_ataques=df_ataques,
        df_perdas=df_perdas,
    )

    # Exportação determinística para data/
    UNIDADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_unidades.to_csv(UNIDADES_CSV, index=False)
    df_ataques.to_csv(ATAQUES_CSV, index=False)
    df_perdas.to_csv(PERDAS_CSV, index=False)
    df_painel.to_csv(PAINEL_TRIMESTRAL_CSV, index=False)
    df_serie.to_csv(SERIE_MENSAL_CSV, index=False)

    logger.info("As 5 bases oficiais (.csv) foram gravadas com sucesso em: %s", UNIDADES_CSV.parent)

    return GeneratedDatasets(
        unidades=df_unidades,
        ataques=df_ataques,
        perdas=df_perdas,
        painel_trimestral=df_painel,
        serie_mensal=df_serie,
        grade_lambda_mensal=df_grade,
    )