"""Módulo de verificação estatística do gerador de Monte Carlo (Questão 2).

Realiza os quatro diagnósticos de verificação exigidos na Questão 2 (Cap. 4, 5 e 12):
    1. Média mensal de ataques observada vs. média teórica de lambda_{u,t} por unidade.
    2. Índice de dispersão Var(N)/E(N) dentro de uma unidade sem SOC vs. todas as
       unidades agrupadas (demonstração da heterogeneidade transversal).
    3. Frequências relativas empíricas dos tipos de ataque vs. probabilidades
       programadas à luz do erro padrão de simulação sigma/sqrt(n).
    4. Taxa geral observada de materialização vs. média teórica dos p_i.
"""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import N_MESES, TIPOS_ATAQUE
from src.parte_a.generator import GeneratedDatasets
from src.utils.logger import get_logger

logger = get_logger(name="parte_a.validator")


def validate_lambda_by_unit(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    validation_dir: Path,
) -> pd.DataFrame:
    """Compara a média mensal observada de ataques com a média teórica de lambda_{u,t}.

    Args:
        datasets: Bases geradas pelo modelo da Parte A.
        tables_dir: Diretório `output/tables/` para exportação CSV.
        validation_dir: Diretório `output/validation/` para exportação de figuras.

    Returns:
        DataFrame com a comparação por unidade e o erro padrão de simulação.
    """
    df_grade = datasets.grade_lambda_mensal

    df_val = (
        df_grade.groupby(["unidade", "soc"], as_index=False)
        .agg(
            lambda_medio_teorico=("lambda_mensal", "mean"),
            media_mensal_observada=("n_ataques", "mean"),
            variancia_mensal_observada=("n_ataques", lambda s: float(np.var(s, ddof=1))),
            total_ataques_36m=("n_ataques", "sum"),
        )
    )

    # Pela teoria de simulação (Seção 12.3), EP(media) = sigma / sqrt(n_meses)
    # Na Poisson, sigma_teorico = sqrt(lambda_medio_teorico)
    df_val["ep_simulacao_teorico"] = np.sqrt(df_val["lambda_medio_teorico"] / N_MESES)
    df_val["ep_amostral"] = np.sqrt(df_val["variancia_mensal_observada"] / N_MESES)
    df_val["diferenca_obs_menos_teo"] = (
        df_val["media_mensal_observada"] - df_val["lambda_medio_teorico"]
    )
    df_val["z_score_erro_simulacao"] = (
        df_val["diferenca_obs_menos_teo"] / df_val["ep_simulacao_teorico"]
    )

    for filename in ("validação_lambda.csv", "validacao_lambda.csv"):
        df_val.to_csv(tables_dir / filename, index=False, float_format="%.4f")

    # Gráfico comparativo Teórico vs. Empírico por unidade
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    x = df_val["unidade"].to_numpy()
    ax.errorbar(
        x,
        df_val["lambda_medio_teorico"],
        yerr=1.96 * df_val["ep_simulacao_teorico"],
        fmt="o",
        color="#1f77b4",
        ecolor="#aec7e8",
        elinewidth=2,
        capsize=4,
        label="Média Teórica Implicada por $\\lambda_{u,t}$ (±1,96 EP)",
    )
    ax.scatter(
        x,
        df_val["media_mensal_observada"],
        color="#d62728",
        marker="x",
        s=65,
        zorder=5,
        label="Média Mensal Observada (36 meses)",
    )
    ax.set_xticks(x)
    ax.set_title("Questão 2-a: Média Mensal de Ataques Observada vs. Teórica por Unidade")
    ax.set_xlabel("Unidade de Negócio da ANDI (1 a 24)")
    ax.set_ylabel("Número Médio de Ataques por Mês")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(validation_dir / "comparacao_lambda_unidades.png")
    plt.close(fig)

    logger.info("Validação de lambda por unidade concluída (Q2-a).")
    return df_val


def analyze_overdispersion(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    validation_dir: Path,
) -> pd.DataFrame:
    """Analisa a razão Var(N)/E(N) dentro de unidades sem SOC vs. todas as unidades (Q2-b).

    Explicação estatística:
        - Dentro de uma única unidade sem SOC, a taxa lambda_{u,t} varia apenas pelos
          fatores temporais suaves (tendência +0,15q, sazonalidade S_m e o pico C_28=1,8),
          mantendo a razão Var(N)/E(N) próxima de 1 (propriedade de equidispersão da Poisson).
        - Quando se juntam todas as 24 unidades em um único conjunto de 864 observações,
          a Lei da Variância Total impõe:
              Var(N) = E[Var(N | lambda)] + Var(E[N | lambda]) = E[lambda] + Var(lambda).
          Como cada unidade tem perfil distinto (servicos, patch, treino, soc), Var(lambda) > 0
          gera forte sobredispersão aparente (Var(N)/E(N) > 1).

    Args:
        datasets: Bases geradas pelo modelo da Parte A.
        tables_dir: Diretório `output/tables/` para exportação CSV.
        validation_dir: Diretório `output/validation/` para exportação de figuras.

    Returns:
        DataFrame com as métricas de dispersão por escopo de agregação.
    """
    df_grade = datasets.grade_lambda_mensal
    df_sem_soc = df_grade[df_grade["soc"] == 0].copy()

    # Seleciona a primeira unidade sem SOC como caso representativo individual
    unidade_ref: int = int(df_sem_soc["unidade"].iloc[0])
    df_u_ref_36m = df_sem_soc[df_sem_soc["unidade"] == unidade_ref]
    df_u_ref_sem_campanha = df_u_ref_36m[df_u_ref_36m["t"] != 28]

    # Calcula Var(N)/E(N) individualmente para cada unidade sem SOC e tira a média intra-unidade
    disp_por_unidade_sem_soc = []
    for u_id, grupo in df_sem_soc.groupby("unidade"):
        media_u = float(grupo["n_ataques"].mean())
        var_u = float(np.var(grupo["n_ataques"], ddof=1))
        disp_por_unidade_sem_soc.append(
            {
                "escopo": f"Unidade {int(u_id):02d} (Sem SOC - 36 meses)",
                "n_observacoes": len(grupo),
                "media_ataques": media_u,
                "variancia_ataques": var_u,
                "razao_var_media": var_u / media_u if media_u > 0 else np.nan,
            }
        )

    df_individuais = pd.DataFrame(disp_por_unidade_sem_soc)

    # Escopos consolidados para comparação direta da Questão 2-b
    media_ref_36 = float(df_u_ref_36m["n_ataques"].mean())
    var_ref_36 = float(np.var(df_u_ref_36m["n_ataques"], ddof=1))

    media_ref_35 = float(df_u_ref_sem_campanha["n_ataques"].mean())
    var_ref_35 = float(np.var(df_u_ref_sem_campanha["n_ataques"], ddof=1))

    media_todas_sem_soc = float(df_sem_soc["n_ataques"].mean())
    var_todas_sem_soc = float(np.var(df_sem_soc["n_ataques"], ddof=1))

    media_global = float(df_grade["n_ataques"].mean())
    var_global = float(np.var(df_grade["n_ataques"], ddof=1))

    resumo_rows = [
        {
            "escopo": f"Unidade Representativa Sem SOC (Unidade {unidade_ref:02d} - 36 meses)",
            "n_observacoes": len(df_u_ref_36m),
            "media_ataques": media_ref_36,
            "variancia_ataques": var_ref_36,
            "razao_var_media": var_ref_36 / media_ref_36,
        },
        {
            "escopo": f"Unidade Representativa Sem SOC (Unidade {unidade_ref:02d} - Sem Campanha t=28)",
            "n_observacoes": len(df_u_ref_sem_campanha),
            "media_ataques": media_ref_35,
            "variancia_ataques": var_ref_35,
            "razao_var_media": var_ref_35 / media_ref_35,
        },
        {
            "escopo": "Média Intra-Unidade (Unidades Sem SOC analisadas separadamente)",
            "n_observacoes": 36,
            "media_ataques": float(df_individuais["media_ataques"].mean()),
            "variancia_ataques": float(df_individuais["variancia_ataques"].mean()),
            "razao_var_media": float(df_individuais["razao_var_media"].mean()),
        },
        {
            "escopo": "Pool de Todas as Unidades Sem SOC Juntas (Heterogeneidade de Perfil)",
            "n_observacoes": len(df_sem_soc),
            "media_ataques": media_todas_sem_soc,
            "variancia_ataques": var_todas_sem_soc,
            "razao_var_media": var_todas_sem_soc / media_todas_sem_soc,
        },
        {
            "escopo": "Pool Global de Todas as 24 Unidades Juntas (864 unidades-mês)",
            "n_observacoes": len(df_grade),
            "media_ataques": media_global,
            "variancia_ataques": var_global,
            "razao_var_media": var_global / media_global,
        },
    ]

    df_sobredisp = pd.concat(
        [pd.DataFrame(resumo_rows), df_individuais], ignore_index=True
    )

    for filename in ("sobredispersão.csv", "sobredispersao.csv"):
        df_sobredisp.to_csv(tables_dir / filename, index=False, float_format="%.4f")

    # Gráfico de barras da Razão Variância / Média (Índice de Dispersão de Fisher)
    df_plot = pd.DataFrame(resumo_rows)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    labels_curtos = [
        f"Unidade {unidade_ref:02d} (s/ SOC)\n[36 meses]",
        f"Unidade {unidade_ref:02d} (s/ SOC)\n[s/ campanha]",
        "Média Intra-Unidade\n[Unidades s/ SOC]",
        "Pool Unidades s/ SOC\n[Juntas]",
        "Pool Global (24 Unid.)\n[Todas Juntas]",
    ]
    valores = df_plot["razao_var_media"].to_numpy()
    cores = ["#2ca02c", "#98df8a", "#1f77b4", "#ff7f0e", "#d62728"]

    bars = ax.bar(labels_curtos, valores, color=cores, edgecolor="black", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Equidispersão Teórica Poisson (Var/E = 1,0)")
    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.04,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Questão 2-b: Índice de Dispersão Var(N)/E(N) — Efeito da Heterogeneidade entre Unidades")
    ax.set_ylabel("Razão Variância / Média")
    ax.set_ylim(0, max(valores) * 1.18)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(validation_dir / "sobredispersao_poisson.png")
    plt.close(fig)

    logger.info("Análise de sobredispersão concluída (Q2-b).")
    return df_sobredisp


def validate_attack_types(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    validation_dir: Path,
) -> pd.DataFrame:
    """Compara frequências relativas dos tipos de ataque com as probabilidades programadas (Q2-c).

    Args:
        datasets: Bases geradas pelo modelo da Parte A.
        tables_dir: Diretório `output/tables/` para exportação CSV.
        validation_dir: Diretório `output/validation/` para exportação de figuras.

    Returns:
        DataFrame comparativo com erro padrão de simulação sigma/sqrt(n).
    """
    df_ataques = datasets.ataques
    n_total: int = len(df_ataques)
    contagens = df_ataques["tipo"].value_counts()

    linhas = []
    for spec in TIPOS_ATAQUE:
        n_obs: int = int(contagens.get(spec.nome, 0))
        freq_rel_obs: float = n_obs / n_total
        p_prog: float = spec.probabilidade

        # Erro padrão de simulação para proporção binomial: sigma / sqrt(n) = sqrt(p*(1-p)/n)
        ep_simulacao: float = float(np.sqrt(p_prog * (1.0 - p_prog) / n_total))
        dif: float = freq_rel_obs - p_prog
        z_score: float = dif / ep_simulacao

        linhas.append(
            {
                "tipo": spec.nome,
                "contagem_observada": n_obs,
                "freq_relativa_observada": freq_rel_obs,
                "probabilidade_programada": p_prog,
                "diferenca_obs_menos_prog": dif,
                "erro_padrao_simulacao": ep_simulacao,
                "ic95_inferior_teorico": p_prog - 1.96 * ep_simulacao,
                "ic95_superior_teorico": p_prog + 1.96 * ep_simulacao,
                "z_score": z_score,
            }
        )

    df_tipos = pd.DataFrame(linhas)

    for filename in ("validação_tipos.csv", "validacao_tipos.csv"):
        df_tipos.to_csv(tables_dir / filename, index=False, float_format="%.6f")

    # Gráfico comparativo de frequências relativas vs. probabilidades programadas
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    x = np.arange(len(df_tipos))
    largura = 0.36

    ax.bar(
        x - largura / 2,
        df_tipos["probabilidade_programada"],
        width=largura,
        yerr=1.96 * df_tipos["erro_padrao_simulacao"],
        capsize=4,
        color="#1f77b4",
        label="Probabilidade Programada (±1,96 $\\sigma/\\sqrt{n}$)",
    )
    ax.bar(
        x + largura / 2,
        df_tipos["freq_relativa_observada"],
        width=largura,
        color="#ff7f0e",
        label=f"Frequência Relativa Observada (n = {n_total})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_tipos["tipo"], rotation=15, ha="right")
    ax.set_ylabel("Proporção / Probabilidade")
    ax.set_title("Questão 2-c: Frequência Relativa dos Tipos de Ataque vs. Probabilidade Programada")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(validation_dir / "comparacao_tipos_ataque.png")
    plt.close(fig)

    logger.info("Validação das frequências dos tipos de ataque concluída (Q2-c).")
    return df_tipos


def validate_materialization_rate(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    validation_dir: Path,
) -> pd.DataFrame:
    """Compara a taxa de materialização observada com a média das probabilidades p_i (Q2-c).

    Args:
        datasets: Bases geradas pelo modelo da Parte A.
        tables_dir: Diretório `output/tables/` para exportação CSV.
        validation_dir: Diretório `output/validation/` para exportação de figuras.

    Returns:
        DataFrame contendo a validação global e estratificada por tipo de ataque.
    """
    df_ataques = datasets.ataques

    def _summarize_segment(df_seg: pd.DataFrame, nome_segmento: str) -> Dict[str, float | str | int]:
        n_seg: int = len(df_seg)
        p_vec: np.ndarray = df_seg["p_materializacao"].to_numpy(dtype=float)
        y_vec: np.ndarray = df_seg["materializou"].to_numpy(dtype=float)

        media_p_teorica: float = float(np.mean(p_vec))
        taxa_observada: float = float(np.mean(y_vec))

        # Erro padrão de simulação (Seção 12.3): DP amostral de Bernoulli / sqrt(n)
        sigma_amostral: float = float(np.std(y_vec, ddof=1))
        ep_simulacao: float = sigma_amostral / float(np.sqrt(n_seg))
        # Erro padrão exato de Poisson-Binomial: sqrt(sum(p_i*(1-p_i))) / n
        ep_exato_bernoulli: float = float(np.sqrt(np.sum(p_vec * (1.0 - p_vec))) / n_seg)

        dif: float = taxa_observada - media_p_teorica
        return {
            "segmento": nome_segmento,
            "n_ataques": n_seg,
            "n_materializados": int(np.sum(y_vec)),
            "media_p_i_programada": media_p_teorica,
            "taxa_materializacao_observada": taxa_observada,
            "diferenca_obs_menos_teo": dif,
            "ep_simulacao_sigma_sqrtn": ep_simulacao,
            "ep_exato_bernoulli": ep_exato_bernoulli,
            "z_score": dif / ep_exato_bernoulli if ep_exato_bernoulli > 0 else 0.0,
        }

    linhas = [_summarize_segment(df_ataques, "Geral (Todos os Ataques)")]
    for spec in TIPOS_ATAQUE:
        df_sub = df_ataques[df_ataques["tipo"] == spec.nome]
        if len(df_sub) > 0:
            linhas.append(_summarize_segment(df_sub, f"Tipo: {spec.nome}"))

    df_mat = pd.DataFrame(linhas)

    for filename in ("validação_materialização.csv", "validacao_materializacao.csv"):
        df_mat.to_csv(tables_dir / filename, index=False, float_format="%.6f")

    # Gráfico comparativo da taxa de materialização observada vs. média dos p_i
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    x = np.arange(len(df_mat))
    largura = 0.36

    ax.bar(
        x - largura / 2,
        df_mat["media_p_i_programada"],
        width=largura,
        yerr=1.96 * df_mat["ep_simulacao_sigma_sqrtn"],
        capsize=4,
        color="#2ca02c",
        label="Média dos $p_i$ Programados (±1,96 $\\sigma/\\sqrt{n}$)",
    )
    ax.bar(
        x + largura / 2,
        df_mat["taxa_materializacao_observada"],
        width=largura,
        color="#9467bd",
        label="Taxa de Materialização Observada",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_mat["segmento"], rotation=18, ha="right")
    ax.set_ylabel("Taxa de Materialização")
    ax.set_title("Questão 2-c: Taxa de Materialização Observada vs. Média Teórica dos $p_i$")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(validation_dir / "comparacao_materializacao.png")
    plt.close(fig)

    logger.info("Validação da taxa de materialização concluída (Q2-c).")
    return df_mat


def run_parte_a_validation(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    validation_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Orquestra todas as verificações da Questão 2 e exporta tabelas e figuras.

    Args:
        datasets: Objeto `GeneratedDatasets` retornado pelo gerador da Parte A.
        tables_dir: Diretório `output/tables/`.
        validation_dir: Diretório `output/validation/`.

    Returns:
        Dicionário contendo os quatro DataFrames de validação da Questão 2.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    df_lambda = validate_lambda_by_unit(datasets, tables_dir, validation_dir)
    df_sobredisp = analyze_overdispersion(datasets, tables_dir, validation_dir)
    df_tipos = validate_attack_types(datasets, tables_dir, validation_dir)
    df_mat = validate_materialization_rate(datasets, tables_dir, validation_dir)

    return {
        "lambda": df_lambda,
        "sobredispersao": df_sobredisp,
        "tipos": df_tipos,
        "materializacao": df_mat,
    }