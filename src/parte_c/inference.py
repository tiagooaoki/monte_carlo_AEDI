"""Módulo de inferência estatística, testes de hipóteses, TCL e regressão simples (Questões 7 a 10).

Implementa os métodos inferenciais da Fase 3 do estudo de caso ANDI:
    - Questão 7 (`analyze_q7_confidence_intervals`): Intervalos de confiança (90%, 95% e 99%)
      via distribuição t de Student para `tempo_deteccao_h` estratificado por `soc` (1 vs. 0).
    - Questão 8 (`analyze_q8_hypothesis_test`): Teste t unilateral à esquerda (H0: mu >= 6 vs.
      H1: mu < 6) para o tempo médio de detecção nas unidades com SOC (`soc == 1`).
    - Questão 9 (`analyze_q9_clt`): Simulação empírica do Teorema Central do Limite (TCL) sobre
      `tempo_deteccao_h` com 1.000 reamostragens para n in {5, 10, 30, 100}.
    - Questão 10 (`analyze_q10_linear_regression`): Regressão linear simples por MQO
      (`perda_rs ~ tempo_deteccao_h`) nos ataques materializados, tabela ANOVA completa
      (SSR, SSE, SST, F = t^2) e gráfico com reta ajustada e banda de IC 95%.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy import stats
import statsmodels.api as sm

from src.config import DEFAULT_ALPHA, SEED
from src.utils.logger import get_logger
from src.utils.rng import get_rng

logger = get_logger(name="parte_c.inference")


def analyze_q7_confidence_intervals(
    df_ataques: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Calcula intervalos de confiança t de Student (90%, 95%, 99%) para `tempo_deteccao_h` por SOC.

    Fórmula aplicada para cada nível de confiança (1 - alpha):
        IC = media +/- t_{gl, 1 - alpha/2} * (s / sqrt(n)), com gl = n - 1.

    Args:
        df_ataques: DataFrame `ataques.csv` contendo os incidentes registrados.
        tables_dir: Diretório `output/tables/` para exportação do CSV.
        figures_dir: Diretório `output/figures/` para exportação do PNG.

    Returns:
        DataFrame contendo n, média, desvio-padrão, erro padrão e ICs de 90%, 95% e 99%.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    grupos: List[Tuple[int, str]] = [
        (1, "Com SOC (SOC = 1)"),
        (0, "Sem SOC (SOC = 0)"),
    ]
    niveis_confianca: Tuple[Tuple[str, float], ...] = (
        ("90", 0.90),
        ("95", 0.95),
        ("99", 0.99),
    )

    linhas: List[Dict[str, float | int | str]] = []

    for soc_val, rotulo in grupos:
        amostra: np.ndarray = df_ataques.loc[
            df_ataques["soc"] == soc_val, "tempo_deteccao_h"
        ].to_numpy(dtype=float)

        n: int = len(amostra)
        gl: int = n - 1
        media: float = float(np.mean(amostra))
        desvio_padrao: float = float(np.std(amostra, ddof=1))
        erro_padrao: float = desvio_padrao / float(np.sqrt(n))

        registro: Dict[str, float | int | str] = {
            "soc": soc_val,
            "grupo": rotulo,
            "n": n,
            "graus_liberdade": gl,
            "media": media,
            "desvio_padrao": desvio_padrao,
            "erro_padrao": erro_padrao,
        }

        for sufixo, nivel in niveis_confianca:
            alpha_ci: float = 1.0 - nivel
            t_crit: float = float(stats.t.ppf(1.0 - alpha_ci / 2.0, df=gl))
            margem: float = t_crit * erro_padrao
            registro[f"t_critico_{sufixo}"] = t_crit
            registro[f"margem_erro_{sufixo}"] = margem
            registro[f"ic{sufixo}_inferior"] = media - margem
            registro[f"ic{sufixo}_superior"] = media + margem

        linhas.append(registro)

    df_q7 = pd.DataFrame(linhas)
    out_csv: Path = tables_dir / "q7_intervalos_confianca.csv"
    df_q7.to_csv(out_csv, index=False, float_format="%.6f")

    # -------------------------------------------------------------------------
    # Figura Q7: Intervalos de Confiança (90%, 95% e 99%) por grupo de SOC
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150, sharey=False)
    cores_niveis = {"90": "#2ca02c", "95": "#1f77b4", "99": "#d62728"}
    deslocamentos_y = {"90": 0.22, "95": 0.0, "99": -0.22}

    for ax, (_, row) in zip(axes, df_q7.iterrows()):
        media_g: float = float(row["media"])
        n_g: int = int(row["n"])
        ep_g: float = float(row["erro_padrao"])

        for sufixo, nivel in niveis_confianca:
            y_pos: float = deslocamentos_y[sufixo]
            inf: float = float(row[f"ic{sufixo}_inferior"])
            sup: float = float(row[f"ic{sufixo}_superior"])
            meia_largura: float = float(row[f"margem_erro_{sufixo}"])

            ax.errorbar(
                x=media_g,
                y=y_pos,
                xerr=meia_largura,
                fmt="o",
                color=cores_niveis[sufixo],
                ecolor=cores_niveis[sufixo],
                elinewidth=2.5,
                capsize=6,
                markersize=6,
                label=f"IC {int(nivel * 100)}%: [{inf:.3f}h; {sup:.3f}h]",
            )

        ax.axvline(
            media_g,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label=f"Média amostral: {media_g:.3f}h (EP = {ep_g:.3f}h)",
        )
        ax.set_yticks([0.22, 0.0, -0.22])
        ax.set_yticklabels(["IC 90%", "IC 95%", "IC 99%"])
        ax.set_ylim(-0.45, 0.45)
        ax.set_xlabel("Tempo Médio de Detecção Estimado (horas)")
        ax.set_title(f"{row['grupo']} (n = {n_g})")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 7: Intervalos de Confiança (t de Student) para o Tempo Médio de Detecção por SOC",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_fig: Path = figures_dir / "q7_intervalos_confianca.png"
    fig.savefig(out_fig)
    plt.close(fig)

    logger.info("Questão 7 concluída: CSV salvo em %s e PNG em %s.", out_csv, out_fig)
    return df_q7


def analyze_q8_hypothesis_test(
    df_ataques: pd.DataFrame,
    tables_dir: Path,
    mu_0: float = 6.0,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Executa teste t unilateral à esquerda (H0: mu >= 6 vs. H1: mu < 6) para SOC = 1.

    Roteiro do teste unilateral inferior (Seção 7.4):
        1. Hipóteses: H0: mu >= 6,0 h vs. H1: mu < 6,0 h.
        2. Estatística de teste: t_calc = (x_barra - mu_0) / (s / sqrt(n)) ~ t_{n-1}.
        3. Nível de significância: alpha = 5% (região crítica: t < t_{n-1; 0,05}).
        4. Cálculo do p-valor unilateral: P(T_{n-1} <= t_calc).
        5. Decisão estatística: Rejeitar H0 se p-valor < alpha.

    Args:
        df_ataques: DataFrame `ataques.csv`.
        tables_dir: Diretório `output/tables/` para exportação do resultado.
        mu_0: Valor de referência sob a hipótese nula (padrão = 6,0 horas).
        alpha: Nível de significância do teste (padrão = 0,05).

    Returns:
        DataFrame contendo as estatísticas do teste t unilateral e a decisão.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    amostra_soc: np.ndarray = df_ataques.loc[
        df_ataques["soc"] == 1, "tempo_deteccao_h"
    ].to_numpy(dtype=float)

    n: int = len(amostra_soc)
    gl: int = n - 1
    media_amostral: float = float(np.mean(amostra_soc))
    desvio_padrao: float = float(np.std(amostra_soc, ddof=1))
    erro_padrao: float = desvio_padrao / float(np.sqrt(n))

    estatistica_t: float = (media_amostral - mu_0) / erro_padrao
    t_critico_unilateral: float = float(stats.t.ppf(alpha, df=gl))
    p_valor: float = float(stats.t.cdf(estatistica_t, df=gl))

    rejeita_h0: bool = bool(p_valor < alpha)
    decisao: str = (
        "Rejeitar H0 (evidência estatística de que mu < 6,0h)"
        if rejeita_h0
        else "Não rejeitar H0 (sem evidência suficiente de que mu < 6,0h)"
    )

    df_q8 = pd.DataFrame(
        [
            {
                "filtro": "SOC == 1",
                "variavel": "tempo_deteccao_h",
                "hipotese_nula_h0": f"mu >= {mu_0:.2f}",
                "hipotese_alternativa_h1": f"mu < {mu_0:.2f}",
                "mu_0_referencia_h": mu_0,
                "alpha": alpha,
                "n": n,
                "graus_liberdade": gl,
                "media_amostral": media_amostral,
                "desvio_padrao_amostral": desvio_padrao,
                "erro_padrao": erro_padrao,
                "estatistica_t": estatistica_t,
                "t_critico_unilateral": t_critico_unilateral,
                "p_valor": p_valor,
                "rejeita_h0": rejeita_h0,
                "decisao": decisao,
            }
        ]
    )

    out_csv: Path = tables_dir / "q8_teste_hipotese_soc.csv"
    df_q8.to_csv(out_csv, index=False, float_format="%.6f")

    logger.info(
        "Questão 8 concluída: t = %.4f, gl = %d, p-valor = %.4e -> %s.",
        estatistica_t,
        gl,
        p_valor,
        decisao,
    )
    return df_q8


def analyze_q9_clt(
    df_ataques: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    rng: Optional[Generator] = None,
    sample_sizes: Tuple[int, ...] = (5, 10, 30, 100),
    n_repetitions: int = 1_000,
) -> pd.DataFrame:
    """Demonstra empiricamente o Teorema Central do Limite (TCL) em `tempo_deteccao_h`.

    Realiza `n_repetitions` reamostragens com reposição para cada tamanho de amostra
    n in {5, 10, 30, 100}, comparando média, desvio-padrão das médias vs. sigma/sqrt(n),
    assimetria e curtose, e exportando os histogramas de convergência.

    Args:
        df_ataques: DataFrame `ataques.csv`.
        tables_dir: Diretório `output/tables/` para exportação do CSV.
        figures_dir: Diretório `output/figures/` para exportação do PNG.
        rng: Instância opcional de `Generator`. Se `None`, instancia `get_rng(SEED)` isolado.
        sample_sizes: Tupla de tamanhos amostrais avaliados (padrão: 5, 10, 30, 100).
        n_repetitions: Número de repetições de Monte Carlo por tamanho amostral (padrão: 1.000).

    Returns:
        DataFrame comparativo das distribuições amostrais das médias.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Usa gerador isolado para não interferir em nenhuma outra etapa do pipeline
    local_rng: Generator = get_rng(SEED) if rng is None else rng

    populacao: np.ndarray = df_ataques["tempo_deteccao_h"].to_numpy(dtype=float)
    media_base: float = float(np.mean(populacao))
    desvio_base: float = float(np.std(populacao, ddof=1))
    assimetria_base: float = float(stats.skew(populacao, bias=False))
    curtose_exc_base: float = float(stats.kurtosis(populacao, fisher=True, bias=False))

    registros_tcl: List[Dict[str, float | int | str]] = [
        {
            "cenario": "Base Original (n = 1 observação)",
            "tamanho_amostra_n": 1,
            "n_repeticoes": len(populacao),
            "media_das_medias": media_base,
            "media_teorica": media_base,
            "desvio_empirico_medias": desvio_base,
            "erro_padrao_teorico": desvio_base,
            "razao_desvio_emp_teo": 1.0,
            "assimetria": assimetria_base,
            "curtose_excesso": curtose_exc_base,
            "curtose_bruta": curtose_exc_base + 3.0,
            "ks_stat_normal": np.nan,
            "ks_pvalor_normal": np.nan,
        }
    ]

    medias_por_n: Dict[int, np.ndarray] = {}

    for n_tam in sample_sizes:
        amostras: np.ndarray = local_rng.choice(
            populacao, size=(n_repetitions, n_tam), replace=True
        )
        medias_amostrais: np.ndarray = np.mean(amostras, axis=1)
        medias_por_n[n_tam] = medias_amostrais

        media_emp: float = float(np.mean(medias_amostrais))
        desvio_emp: float = float(np.std(medias_amostrais, ddof=1))
        ep_teorico: float = desvio_base / float(np.sqrt(n_tam))
        skew_emp: float = float(stats.skew(medias_amostrais, bias=False))
        kurt_exc_emp: float = float(
            stats.kurtosis(medias_amostrais, fisher=True, bias=False)
        )
        kurt_bruta_emp: float = float(
            stats.kurtosis(medias_amostrais, fisher=False, bias=False)
        )

        ks_stat, ks_pval = stats.ks_1samp(
            medias_amostrais,
            stats.norm(loc=media_base, scale=ep_teorico).cdf,
        )

        registros_tcl.append(
            {
                "cenario": f"Médias Amostrais (n = {n_tam})",
                "tamanho_amostra_n": n_tam,
                "n_repeticoes": n_repetitions,
                "media_das_medias": media_emp,
                "media_teorica": media_base,
                "desvio_empirico_medias": desvio_emp,
                "erro_padrao_teorico": ep_teorico,
                "razao_desvio_emp_teo": desvio_emp / ep_teorico,
                "assimetria": skew_emp,
                "curtose_excesso": kurt_exc_emp,
                "curtose_bruta": kurt_bruta_emp,
                "ks_stat_normal": float(ks_stat),
                "ks_pvalor_normal": float(ks_pval),
            }
        )

    df_q9 = pd.DataFrame(registros_tcl)
    out_csv: Path = tables_dir / "q9_tcl_simulacao.csv"
    df_q9.to_csv(out_csv, index=False, float_format="%.6f")

    # -------------------------------------------------------------------------
    # Figura Q9: Histogramas de convergência do TCL (n = 5, 10, 30, 100)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=150)
    axes_flat = axes.flatten()
    cores_painel = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    for idx, n_tam in enumerate(sample_sizes):
        ax = axes_flat[idx]
        vetor_medias = medias_por_n[n_tam]
        linha_res = df_q9[df_q9["tamanho_amostra_n"] == n_tam].iloc[0]

        media_m: float = float(linha_res["media_das_medias"])
        dp_m: float = float(linha_res["desvio_empirico_medias"])
        ep_teo: float = float(linha_res["erro_padrao_teorico"])
        skew_m: float = float(linha_res["assimetria"])
        kurt_m: float = float(linha_res["curtose_excesso"])

        ax.hist(
            vetor_medias,
            bins=30,
            density=True,
            color=cores_painel[idx],
            edgecolor="black",
            alpha=0.72,
            label=f"Empírico ({n_repetitions} médias)",
        )

        x_grid = np.linspace(
            float(np.min(vetor_medias)) * 0.9,
            float(np.max(vetor_medias)) * 1.05,
            300,
        )
        pdf_norm = stats.norm.pdf(x_grid, loc=media_base, scale=ep_teo)
        ax.plot(
            x_grid,
            pdf_norm,
            "r--",
            linewidth=2.2,
            label=f"Normal Teórica ($\\mu={media_base:.2f}, \\sigma/\\sqrt{{n}}={ep_teo:.2f}$)",
        )
        ax.axvline(
            media_m,
            color="black",
            linestyle=":",
            linewidth=1.6,
            label=f"Média das Médias = {media_m:.2f}h",
        )

        ax.set_title(
            f"Tamanho da Amostra $n = {n_tam}$\n"
            f"DP Emp. = {dp_m:.3f} (Teo. = {ep_teo:.3f}) | Assim. = {skew_m:.2f} | Curt.(exc.) = {kurt_m:.2f}"
        )
        ax.set_xlabel("Média Amostral de `tempo_deteccao_h` (horas)")
        ax.set_ylabel("Densidade")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 9: Convergência Empírica do Teorema Central do Limite (TCL) — `tempo_deteccao_h` (1.000 Repetições)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    out_fig: Path = figures_dir / "q9_tcl_histogramas.png"
    fig.savefig(out_fig)
    plt.close(fig)

    logger.info("Questão 9 concluída: CSV salvo em %s e PNG em %s.", out_csv, out_fig)
    return df_q9


def analyze_q10_linear_regression(
    df_perdas: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, pd.DataFrame]:
    """Ajusta a regressão linear simples `perda_rs ~ tempo_deteccao_h` e executa a ANOVA.

    Utiliza exclusivamente os ataques materializados (`perdas.csv`), estimando
    intercepto, coeficiente angular, erros padrões, estatísticas t, p-valores,
    IC 95%, R², decomposição de soma de quadrados (SSR, SSE, SST) e estatística F.

    Args:
        df_perdas: DataFrame `perdas.csv` (ataques materializados).
        tables_dir: Diretório `output/tables/` para exportação dos CSVs.
        figures_dir: Diretório `output/figures/` para exportação do gráfico PNG.
        alpha: Nível de significância para os intervalos de confiança (padrão = 0,05).

    Returns:
        Dicionário contendo os DataFrames `regressao` e `anova`.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    x_tempo: np.ndarray = df_perdas["tempo_deteccao_h"].to_numpy(dtype=float)
    y_perda: np.ndarray = df_perdas["perda_rs"].to_numpy(dtype=float)
    n_obs: int = len(df_perdas)

    X_design: np.ndarray = sm.add_constant(x_tempo)
    modelo = sm.OLS(y_perda, X_design).fit()

    nomes_termos = ["intercepto", "tempo_deteccao_h"]
    ic_matriz: np.ndarray = modelo.conf_int(alpha=alpha)

    r2: float = float(modelo.rsquared)
    r2_adj: float = float(modelo.rsquared_adj)
    ep_regressao: float = float(np.sqrt(modelo.mse_resid))
    correlacao_r: float = float(stats.pearsonr(x_tempo, y_perda)[0])

    linhas_reg: List[Dict[str, float | int | str]] = []
    for idx, termo in enumerate(nomes_termos):
        linhas_reg.append(
            {
                "modelo": "perda_rs ~ tempo_deteccao_h",
                "termo": termo,
                "coeficiente": float(modelo.params[idx]),
                "erro_padrao": float(modelo.bse[idx]),
                "t": float(modelo.tvalues[idx]),
                "p": float(modelo.pvalues[idx]),
                "ic95_inferior": float(ic_matriz[idx, 0]),
                "ic95_superior": float(ic_matriz[idx, 1]),
                "r2": r2,
                "r2_ajustado": r2_adj,
                "correlacao_pearson_r": correlacao_r,
                "erro_padrao_regressao": ep_regressao,
                "n_observacoes": n_obs,
            }
        )

    df_reg = pd.DataFrame(linhas_reg)
    out_reg_csv: Path = tables_dir / "q10_regressao_linear.csv"
    df_reg.to_csv(out_reg_csv, index=False, float_format="%.6f")

    # -------------------------------------------------------------------------
    # Tabela ANOVA: SSR (Regressão), SSE (Resíduos), SST (Total), F e p-valor
    # -------------------------------------------------------------------------
    ssr: float = float(modelo.ess)
    sse: float = float(modelo.ssr)
    sst: float = float(modelo.centered_tss)

    gl_reg: int = int(modelo.df_model)
    gl_res: int = int(modelo.df_resid)
    gl_tot: int = gl_reg + gl_res

    msr: float = ssr / float(gl_reg)
    mse: float = sse / float(gl_res)
    mst: float = sst / float(gl_tot)

    f_stat: float = float(modelo.fvalue)
    f_pval: float = float(modelo.f_pvalue)
    t_slope_quadrado: float = float(modelo.tvalues[1] ** 2)

    df_anova = pd.DataFrame(
        [
            {
                "fonte_variacao": "Regressao (SSR)",
                "sigla": "SSR",
                "graus_liberdade": gl_reg,
                "soma_quadrados": ssr,
                "quadrado_medio": msr,
                "F": f_stat,
                "t_quadrado_beta1": t_slope_quadrado,
                "p": f_pval,
            },
            {
                "fonte_variacao": "Residuo (SSE)",
                "sigla": "SSE",
                "graus_liberdade": gl_res,
                "soma_quadrados": sse,
                "quadrado_medio": mse,
                "F": np.nan,
                "t_quadrado_beta1": np.nan,
                "p": np.nan,
            },
            {
                "fonte_variacao": "Total (SST)",
                "sigla": "SST",
                "graus_liberdade": gl_tot,
                "soma_quadrados": sst,
                "quadrado_medio": mst,
                "F": np.nan,
                "t_quadrado_beta1": np.nan,
                "p": np.nan,
            },
        ]
    )
    out_anova_csv: Path = tables_dir / "q10_anova.csv"
    df_anova.to_csv(out_anova_csv, index=False, float_format="%.6f")

    # -------------------------------------------------------------------------
    # Figura Q10: Scatterplot + Reta Ajustada + Banda de IC 95%
    # -------------------------------------------------------------------------
    ordem_idx = np.argsort(x_tempo)
    x_grid = np.linspace(float(np.min(x_tempo)), float(np.max(x_tempo)), 250)
    X_grid_design = sm.add_constant(x_grid)
    pred_frame = modelo.get_prediction(X_grid_design).summary_frame(alpha=alpha)

    b0: float = float(modelo.params[0])
    b1: float = float(modelo.params[1])

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=150)
    ax.scatter(
        x_tempo,
        y_perda / 1e3,
        alpha=0.45,
        s=24,
        color="#1f77b4",
        edgecolor="none",
        label=f"Incidentes Materializados (n = {n_obs})",
    )
    ax.plot(
        x_grid,
        pred_frame["mean"].to_numpy() / 1e3,
        color="#d62728",
        linewidth=2.3,
        label=f"Reta Ajustada MQO: $\\hat{{y}} = {b0:,.0f} + {b1:,.0f}\\cdot x$ ($R^2 = {r2:.4f}$)",
    )
    ax.fill_between(
        x_grid,
        pred_frame["mean_ci_lower"].to_numpy() / 1e3,
        pred_frame["mean_ci_upper"].to_numpy() / 1e3,
        color="#d62728",
        alpha=0.22,
        label="Intervalo de Confiança de 95% (Média Condicional)",
    )

    ax.set_title(
        f"Questão 10: Regressão Linear Simples (`perda_rs ~ tempo_deteccao_h`)\n"
        f"Teste ANOVA $F = {f_stat:.2f}$ ($t^2 = {t_slope_quadrado:.2f}$), p-valor = {f_pval:.4e}"
    )
    ax.set_xlabel("Tempo até a Detecção `tempo_deteccao_h` (horas)")
    ax.set_ylabel("Perda Financeira `perda_rs` (R$ mil)")
    ax.legend(loc="upper left", fontsize=9.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    out_fig: Path = figures_dir / "q10_regressao_perda_tempo.png"
    fig.savefig(out_fig)
    plt.close(fig)

    logger.info(
        "Questão 10 concluída: CSVs salvos em %s e %s; PNG salvo em %s.",
        out_reg_csv,
        out_anova_csv,
        out_fig,
    )
    return {
        "regressao": df_reg,
        "anova": df_anova,
    }