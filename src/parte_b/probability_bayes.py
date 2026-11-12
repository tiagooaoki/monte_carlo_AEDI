"""Módulo de probabilidade condicional, Teorema de Bayes e ajuste de distribuições (Questões 5 e 6).

Implementa as análises exigidas nas Questões 5 e 6 (Capítulos 3, 4 e 5):
    - Questão 5: Tabela de contingência SOC x Materialização, cálculo das
      probabilidades marginais e condicionais P(Materializou), P(Materializou | SOC),
      P(Materializou | Sem SOC), teste formal de independência e aplicação do
      Teorema de Bayes para P(SOC | Materializou), além das tabelas de contingência
      por tipo de ataque e da regra de triagem (alarme se cvss >= 7).
    - Questão 6: Estimação de parâmetros por Máxima Verossimilhança, teste de
      aderência de Kolmogorov-Smirnov (KS) e gráficos Q-Q (Quantil-Quantil) para
      `tempo_deteccao_h` (Exponencial) e `ln(perda_rs)` (Normal / Lognormal).
"""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.parte_a.generator import GeneratedDatasets
from src.utils.logger import get_logger

logger = get_logger(name="parte_b.probability_bayes")


def analyze_q5_bayes(
    df_ataques: pd.DataFrame,
    tables_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Executa a Questão 5: tabela SOC x Materialização, Teorema de Bayes e Triagem.

    Fórmula de Bayes aplicada:
        P(SOC = 1 | Materializou = 1) =
            [P(Materializou = 1 | SOC = 1) * P(SOC = 1)] / P(Materializou = 1)

    Args:
        df_ataques: DataFrame `ataques.csv` gerado na Parte A.
        tables_dir: Diretório `output/tables/` para exportação de tabelas CSV.

    Returns:
        Dicionário contendo os DataFrames de resultados de Bayes e contingência.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_total: int = len(df_ataques)
    mask_soc = df_ataques["soc"] == 1
    mask_sem_soc = df_ataques["soc"] == 0
    mask_mat = df_ataques["materializou"] == 1
    mask_nao_mat = df_ataques["materializou"] == 0

    # Contagens absolutas da tabela de contingência 2x2 (SOC x Materialização)
    n_soc_mat: int = int(np.sum(mask_soc & mask_mat))
    n_soc_nao_mat: int = int(np.sum(mask_soc & mask_nao_mat))
    n_sem_soc_mat: int = int(np.sum(mask_sem_soc & mask_mat))
    n_sem_soc_nao_mat: int = int(np.sum(mask_sem_soc & mask_nao_mat))

    n_soc_total: int = n_soc_mat + n_soc_nao_mat
    n_sem_soc_total: int = n_sem_soc_mat + n_sem_soc_nao_mat
    n_mat_total: int = n_soc_mat + n_sem_soc_mat
    n_nao_mat_total: int = n_soc_nao_mat + n_sem_soc_nao_mat

    # Probabilidades marginais
    p_soc: float = n_soc_total / n_total
    p_sem_soc: float = n_sem_soc_total / n_total
    p_materializou: float = n_mat_total / n_total
    p_nao_materializou: float = n_nao_mat_total / n_total

    # Probabilidades conjuntas
    p_conj_soc_mat: float = n_soc_mat / n_total
    p_conj_sem_soc_mat: float = n_sem_soc_mat / n_total

    # Teste de independência: P(SOC ∩ Materializou) vs P(SOC) * P(Materializou)
    prod_marginais_soc_mat: float = p_soc * p_materializou
    diferenca_independencia: float = p_conj_soc_mat - prod_marginais_soc_mat

    # Probabilidades condicionais diretas (Verossimilhanças)
    p_mat_dado_soc: float = n_soc_mat / n_soc_total
    p_mat_dado_sem_soc: float = n_sem_soc_mat / n_sem_soc_total

    # Aplicação do Teorema de Bayes: P(SOC | Materializou)
    numerador_bayes: float = p_mat_dado_soc * p_soc
    denominador_prob_total: float = (
        p_mat_dado_soc * p_soc + p_mat_dado_sem_soc * p_sem_soc
    )
    p_soc_dado_mat_bayes: float = numerador_bayes / denominador_prob_total
    p_soc_dado_mat_contagem: float = n_soc_mat / n_mat_total
    erro_absoluto_bayes: float = abs(p_soc_dado_mat_bayes - p_soc_dado_mat_contagem)

    # Risco Relativo e Razão de Chances (Odds Ratio) empírica do SOC
    risco_relativo_soc: float = p_mat_dado_soc / p_mat_dado_sem_soc
    reducao_relativa_risco_pct: float = 100.0 * (1.0 - risco_relativo_soc)

    df_bayes_soc = pd.DataFrame(
        [
            {
                "metrica": "N Total de Ataques",
                "valor": float(n_total),
                "descricao": "Total de ataques registrados nos 36 meses",
            },
            {
                "metrica": "Contagem (SOC=1, Materializou=1)",
                "valor": float(n_soc_mat),
                "descricao": "Ataques em unidades com SOC que geraram perda",
            },
            {
                "metrica": "Contagem (SOC=1, Materializou=0)",
                "valor": float(n_soc_nao_mat),
                "descricao": "Ataques em unidades com SOC contidos sem perda",
            },
            {
                "metrica": "Contagem (SOC=0, Materializou=1)",
                "valor": float(n_sem_soc_mat),
                "descricao": "Ataques em unidades sem SOC que geraram perda",
            },
            {
                "metrica": "Contagem (SOC=0, Materializou=0)",
                "valor": float(n_sem_soc_nao_mat),
                "descricao": "Ataques em unidades sem SOC contidos sem perda",
            },
            {
                "metrica": "P(SOC)",
                "valor": p_soc,
                "descricao": "Probabilidade marginal do ataque ocorrer em unidade com SOC",
            },
            {
                "metrica": "P(Sem SOC)",
                "valor": p_sem_soc,
                "descricao": "Probabilidade marginal do ataque ocorrer em unidade sem SOC",
            },
            {
                "metrica": "P(Materializou)",
                "valor": p_materializou,
                "descricao": "Taxa marginal global de materialização em perda (Prevalência)",
            },
            {
                "metrica": "P(Materializou | SOC)",
                "valor": p_mat_dado_soc,
                "descricao": "Probabilidade condicional de materialização dado que a unidade possui SOC",
            },
            {
                "metrica": "P(Materializou | Sem SOC)",
                "valor": p_mat_dado_sem_soc,
                "descricao": "Probabilidade condicional de materialização dado que a unidade NÃO possui SOC",
            },
            {
                "metrica": "P(SOC ∩ Materializou)",
                "valor": p_conj_soc_mat,
                "descricao": "Probabilidade conjunta observada de ter SOC e materializar",
            },
            {
                "metrica": "P(SOC) * P(Materializou)",
                "valor": prod_marginais_soc_mat,
                "descricao": "Produto das marginais sob hipótese de independência",
            },
            {
                "metrica": "Diferença Independência [P(A∩B) - P(A)P(B)]",
                "valor": diferenca_independencia,
                "descricao": "Diferença negativa confirma dependência (SOC reduz materialização)",
            },
            {
                "metrica": "P(SOC | Materializou) [Teorema de Bayes]",
                "valor": p_soc_dado_mat_bayes,
                "descricao": "Probabilidade a posteriori de um ataque materializado vir de unidade com SOC",
            },
            {
                "metrica": "P(SOC | Materializou) [Contagem Direta]",
                "valor": p_soc_dado_mat_contagem,
                "descricao": "Conferência direta n(SOC=1, Mat=1) / n(Mat=1)",
            },
            {
                "metrica": "Erro Absoluto Bayes vs Contagem",
                "valor": erro_absoluto_bayes,
                "descricao": "Diferença numérica entre a fórmula de Bayes e a contagem na base",
            },
            {
                "metrica": "Risco Relativo RR = P(Mat|SOC) / P(Mat|Sem SOC)",
                "valor": risco_relativo_soc,
                "descricao": "Razão entre as taxas de materialização com e sem SOC",
            },
            {
                "metrica": "Redução Relativa de Risco pelo SOC (%)",
                "valor": reducao_relativa_risco_pct,
                "descricao": "Redução percentual na taxa de materialização associada à presença do SOC",
            },
        ]
    )

    out_bayes_path: Path = tables_dir / "q5_bayes_soc_materializacao.csv"
    df_bayes_soc.to_csv(out_bayes_path, index=False, float_format="%.6f")

    # -------------------------------------------------------------------------
    # Tabelas complementares da Questão 5 (Q5-a: tipo x materializou; Q5-c/d: alarme cvss >= 7)
    # -------------------------------------------------------------------------
    ct_tipo = pd.crosstab(df_ataques["tipo"], df_ataques["materializou"])
    ct_tipo.columns = ["nao_materializou_0", "materializou_1"]
    ct_tipo["total_tipo"] = ct_tipo["nao_materializou_0"] + ct_tipo["materializou_1"]
    ct_tipo["p_marginal_tipo"] = ct_tipo["total_tipo"] / n_total
    ct_tipo["p_conjunta_tipo_e_mat"] = ct_tipo["materializou_1"] / n_total
    ct_tipo["p_mat_dado_tipo"] = ct_tipo["materializou_1"] / ct_tipo["total_tipo"]
    ct_tipo["p_tipo_dado_mat"] = ct_tipo["materializou_1"] / n_mat_total
    ct_tipo = ct_tipo.reset_index()
    ct_tipo.to_csv(
        tables_dir / "q5_contingencia_tipo_materializacao.csv",
        index=False,
        float_format="%.6f",
    )

    # Regra de triagem do SOC: alarme se cvss >= 7.0
    alarme = df_ataques["cvss"] >= 7.0
    sensibilidade: float = float(np.sum(alarme & mask_mat) / n_mat_total)
    especificidade: float = float(np.sum((~alarme) & mask_nao_mat) / n_nao_mat_total)
    taxa_falso_positivo: float = 1.0 - especificidade

    p_alarme_total: float = (
        sensibilidade * p_materializou + taxa_falso_positivo * p_nao_materializou
    )
    p_mat_dado_alarme_bayes: float = (sensibilidade * p_materializou) / p_alarme_total
    p_mat_dado_alarme_direto: float = float(
        np.sum(alarme & mask_mat) / np.sum(alarme)
    )

    # Simulação Q5-d: prevalência caindo pela metade
    prev_metade: float = 0.5 * p_materializou
    vpp_prev_metade: float = (sensibilidade * prev_metade) / (
        sensibilidade * prev_metade + taxa_falso_positivo * (1.0 - prev_metade)
    )

    df_triagem = pd.DataFrame(
        [
            {
                "prevalencia_p_mat": p_materializou,
                "sensibilidade_p_alarme_dado_mat": sensibilidade,
                "especificidade_p_sem_alarme_dado_nao_mat": especificidade,
                "vpp_p_mat_dado_alarme_bayes": p_mat_dado_alarme_bayes,
                "vpp_p_mat_dado_alarme_contagem": p_mat_dado_alarme_direto,
                "prevalencia_reduzida_metade": prev_metade,
                "vpp_com_prevalencia_metade": vpp_prev_metade,
            }
        ]
    )
    df_triagem.to_csv(
        tables_dir / "q5_triagem_cvss_bayes.csv", index=False, float_format="%.6f"
    )

    logger.info("Questão 5 concluída: tabela Bayes salva em %s.", out_bayes_path)
    return {
        "bayes_soc": df_bayes_soc,
        "contingencia_tipo": ct_tipo,
        "triagem_cvss": df_triagem,
    }


def analyze_q6_distributions(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Executa a Questão 6: ajuste de distribuições (Exponencial e Lognormal), KS e Q-Q plots.

    Args:
        datasets: Contêiner `GeneratedDatasets` com as bases da Parte A.
        tables_dir: Diretório `output/tables/` para exportação de tabelas CSV.
        figures_dir: Diretório `output/figures/` para exportação das figuras PNG.

    Returns:
        DataFrame consolidado com parâmetros estimados, testes KS e probabilidades.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_ataques = datasets.ataques
    df_perdas = datasets.perdas

    registros_q6 = []

    # -------------------------------------------------------------------------
    # 1. Ajuste da Distribuição Exponencial para `tempo_deteccao_h`
    # -------------------------------------------------------------------------
    subconjuntos_tempo = [
        (
            "tempo_deteccao_h [Geral - Mistura de Regimes]",
            df_ataques["tempo_deteccao_h"].to_numpy(dtype=float),
            np.nan,
        ),
        (
            "tempo_deteccao_h [Com SOC e No Expediente (theta=4h)]",
            df_ataques.loc[
                (df_ataques["soc"] == 1) & (df_ataques["fora_expediente"] == 0),
                "tempo_deteccao_h",
            ].to_numpy(dtype=float),
            4.0,
        ),
        (
            "tempo_deteccao_h [Com SOC - Todos Horários (E[T]=4,7h)]",
            df_ataques.loc[
                df_ataques["soc"] == 1, "tempo_deteccao_h"
            ].to_numpy(dtype=float),
            4.7,
        ),
        (
            "tempo_deteccao_h [Sem SOC e No Expediente (theta=12h)]",
            df_ataques.loc[
                (df_ataques["soc"] == 0) & (df_ataques["fora_expediente"] == 0),
                "tempo_deteccao_h",
            ].to_numpy(dtype=float),
            12.0,
        ),
    ]

    for nome_esc, vetor_t, theta_prog in subconjuntos_tempo:
        n_obs: int = len(vetor_t)
        theta_hat: float = float(np.mean(vetor_t))
        dp_obs: float = float(np.std(vetor_t, ddof=1))
        ks_stat, ks_pval = stats.ks_1samp(
            vetor_t, stats.expon(loc=0.0, scale=theta_hat).cdf
        )
        # Probabilidade P(T > 8h) teórica vs observada (item Q6-b)
        prob_teo_gt8: float = float(np.exp(-8.0 / theta_hat))
        prob_prog_gt8: float = (
            float(np.exp(-8.0 / theta_prog)) if not np.isnan(theta_prog) else np.nan
        )
        prop_obs_gt8: float = float(np.mean(vetor_t > 8.0))

        registros_q6.append(
            {
                "variavel_escopo": nome_esc,
                "distribuicao_testada": "Exponencial(theta)",
                "n": n_obs,
                "parametro_1_nome": "theta_hat (media h)",
                "parametro_1_estimado": theta_hat,
                "parametro_1_programado": theta_prog,
                "parametro_2_nome": "desvio_padrao_h",
                "parametro_2_estimado": dp_obs,
                "ks_estatistica_d": float(ks_stat),
                "ks_p_valor": float(ks_pval),
                "aderencia_alfa_5pct": "Sim (p >= 0,05)" if ks_pval >= 0.05 else "Não (Mistura/Rejeita)",
                "metrica_cauda_nome": "P(tempo > 8h)",
                "metrica_cauda_teorica_estimada": prob_teo_gt8,
                "metrica_cauda_teorica_programada": prob_prog_gt8,
                "metrica_cauda_empirica_observada": prop_obs_gt8,
            }
        )

    # -------------------------------------------------------------------------
    # 2. Ajuste Normal sobre `ln(perda_rs)` (Hipótese Lognormal para `perda_rs`)
    # -------------------------------------------------------------------------
    subconjuntos_perda = [
        (
            "ln(perda_rs) [Geral - Todas as Perdas]",
            np.log(df_perdas["perda_rs"].to_numpy(dtype=float)),
        ),
        (
            "ln(perda_rs) [Tipo: Ransomware]",
            np.log(
                df_perdas.loc[
                    df_perdas["tipo"] == "Ransomware", "perda_rs"
                ].to_numpy(dtype=float)
            ),
        ),
        (
            "ln(perda_rs) [Tipo: Phishing]",
            np.log(
                df_perdas.loc[
                    df_perdas["tipo"] == "Phishing", "perda_rs"
                ].to_numpy(dtype=float)
            ),
        ),
    ]

    for nome_esc, vetor_ln in subconjuntos_perda:
        n_obs = len(vetor_ln)
        mu_hat: float = float(np.mean(vetor_ln))
        sigma_hat: float = float(np.std(vetor_ln, ddof=1))
        ks_stat, ks_pval = stats.ks_1samp(
            vetor_ln, stats.norm(loc=mu_hat, scale=sigma_hat).cdf
        )
        max_ln: float = float(np.max(vetor_ln))
        z_max: float = (max_ln - mu_hat) / sigma_hat

        registros_q6.append(
            {
                "variavel_escopo": nome_esc,
                "distribuicao_testada": "Normal sobre ln(perda_rs)",
                "n": n_obs,
                "parametro_1_nome": "mu_hat (media ln)",
                "parametro_1_estimado": mu_hat,
                "parametro_1_programado": np.nan,
                "parametro_2_nome": "sigma_hat (dp ln)",
                "parametro_2_estimado": sigma_hat,
                "ks_estatistica_d": float(ks_stat),
                "ks_p_valor": float(ks_pval),
                "aderencia_alfa_5pct": "Sim (p >= 0,05)" if ks_pval >= 0.05 else "Não (p < 0,05)",
                "metrica_cauda_nome": "Z-score da Maior Perda",
                "metrica_cauda_teorica_estimada": z_max,
                "metrica_cauda_teorica_programada": np.nan,
                "metrica_cauda_empirica_observada": float(np.exp(max_ln)),
            }
        )

    df_q6 = pd.DataFrame(registros_q6)
    out_q6_path: Path = tables_dir / "q6_ajuste_distribuicoes.csv"
    df_q6.to_csv(out_q6_path, index=False, float_format="%.6f")

    # -------------------------------------------------------------------------
    # Figura 1 da Q6: q6_qq_exponencial.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=150)

    t_global = df_ataques["tempo_deteccao_h"].to_numpy(dtype=float)
    theta_global = float(np.mean(t_global))
    ks_g_stat, ks_g_pval = stats.ks_1samp(
        t_global, stats.expon(scale=theta_global).cdf
    )
    pp_global = sm.ProbPlot(t_global, stats.expon, scale=theta_global)
    pp_global.qqplot(
        line="45",
        ax=axes[0],
        color="#1f77b4",
        alpha=0.5,
         markersize=3.5,
    )
    axes[0].set_title(
        f"(a) Q-Q Exponencial Global (Estimativa amostral: $\\hat{{\\theta}} = {theta_global:.2f}$ h, $n={len(t_global)}$)\n"
        f"Mistura SOC/Horário: KS $D = {ks_g_stat:.4f}$, $p = {ks_g_pval:.4e}$"
    )
    axes[0].set_xlabel("Quantis Teóricos Exponencial($\\hat{\\theta}$)")
    axes[0].set_ylabel("Quantis Empíricos `tempo_deteccao_h` (h)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    t_soc_exp0 = df_ataques.loc[
        (df_ataques["soc"] == 1) & (df_ataques["fora_expediente"] == 0),
        "tempo_deteccao_h",
    ].to_numpy(dtype=float)
    theta_soc0 = float(np.mean(t_soc_exp0))
    ks_s_stat, ks_s_pval = stats.ks_1samp(
        t_soc_exp0, stats.expon(scale=theta_soc0).cdf
    )
    pp_soc0 = sm.ProbPlot(t_soc_exp0, stats.expon, scale=theta_soc0)
    pp_soc0.qqplot(
        line="45",
        ax=axes[1],
        color="#2ca02c",
        alpha=0.6,
        markersize=4.0,
    )
    axes[1].set_title(
        f"(b) Q-Q Exponencial Homogêneo [Com SOC e No Expediente]\n"
        f"Estimativa amostral: $\\hat{{\\theta}} = {theta_soc0:.2f}$ h (Parâmetro programado da simulação: 4,0 h) | KS $D = {ks_s_stat:.4f}$, $p = {ks_s_pval:.4f}$"
    )
    axes[1].set_xlabel("Quantis Teóricos Exponencial($\\hat{\\theta}$)")
    axes[1].set_ylabel("Quantis Empíricos `tempo_deteccao_h` (h)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 6: Gráfico Q-Q Exponencial para `tempo_deteccao_h` (Mistura Global vs. Regime Homogêneo)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    qq_exp_path: Path = figures_dir / "q6_qq_exponencial.png"
    fig.savefig(qq_exp_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figura 2 da Q6: q6_qq_lognormal.png (Q-Q Normal de ln(perda_rs))
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=150)

    paineis_ln = [
        ("Geral (Todas as Perdas)", subconjuntos_perda[0][1], "#1f77b4"),
        ("Tipo: Ransomware", subconjuntos_perda[1][1], "#d62728"),
        ("Tipo: Phishing", subconjuntos_perda[2][1], "#2ca02c"),
    ]

    for ax, (titulo_p, dados_ln, cor) in zip(axes, paineis_ln):
        mu_p = float(np.mean(dados_ln))
        sd_p = float(np.std(dados_ln, ddof=1))
        ks_d, ks_p = stats.ks_1samp(dados_ln, stats.norm(loc=mu_p, scale=sd_p).cdf)
        z_extremo = (float(np.max(dados_ln)) - mu_p) / sd_p

        pp_ln = sm.ProbPlot(dados_ln, stats.norm, loc=mu_p, scale=sd_p)
        pp_ln.qqplot(line="45", ax=ax, color=cor, alpha=0.65, markersize=4.0)
        ax.set_title(
            f"{titulo_p} ($n={len(dados_ln)}$)\n"
            f"$\\hat{{\\mu}}={mu_p:.2f}, \\hat{{\\sigma}}={sd_p:.2f}$ | KS $p={ks_p:.3f}$ | $Z_{{max}}={z_extremo:.2f}$"
        )
        ax.set_xlabel("Quantis Teóricos Normal($\\hat{\\mu}, \\hat{\\sigma}$)")
        ax.set_ylabel("Quantis Empíricos $\\ln(\\text{perda\\_rs})$")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 6-c: Gráficos Q-Q de $\\ln(\\text{perda\\_rs})$ contra a Normal (Validação da Hipótese Lognormal)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    qq_log_path: Path = figures_dir / "q6_qq_lognormal.png"
    fig.savefig(qq_log_path)
    plt.close(fig)

    logger.info(
        "Questão 6 concluída: tabela salva em %s e Q-Q plots salvos em %s e %s.",
        out_q6_path,
        qq_exp_path,
        qq_log_path,
    )
    return df_q6