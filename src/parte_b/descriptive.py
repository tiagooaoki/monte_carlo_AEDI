"""Módulo de estatística descritiva, quantis e visualização (Questões 3 e 4).

Implementa as análises exigidas nas Questões 3 e 4 (Capítulos 1 e 2):
    - Questão 3: Estatísticas descritivas completas de `perda_rs` (n, média, mediana,
      moda aproximada, mínimo, máximo, desvio-padrão, CV, AIQ, assimetria, curtose e
      percentis P50, P90, P95, P99), tabela de frequências pela Regra de Sturges e
      histogramas comparativos (Regra de Sturges vs. Regra de Freedman-Diaconis e
      escala logarítmica).
    - Questão 4: Box-plots de `perda_rs` por tipo de ataque com identificação de
      outliers pela regra 1,5 x AIQ, mapa de calor (heatmap) de correlações,
      scatterplots (patch x materializou, treino x materializou, cvss x perda_rs),
      matrizes do painel trimestral e VaR histórico mensal de 95%.
"""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import SEED, TIPOS_ATAQUE
from src.parte_a.generator import GeneratedDatasets
from src.utils.logger import get_logger

logger = get_logger(name="parte_b.descriptive")


def _calculate_sturges_bins(n: int) -> int:
    """Calcula o número de classes pela Regra de Sturges: k = ceil(1 + 3.322 * log10(n))."""
    if n <= 1:
        return 1
    return int(np.ceil(1.0 + 3.322 * np.log10(float(n))))


def _calculate_freedman_diaconis_bins(values: np.ndarray) -> int:
    """Calcula o número de classes pela Regra de Freedman-Diaconis: h = 2 * AIQ * n^(-1/3)."""
    n: int = len(values)
    if n <= 1:
        return 1
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr: float = float(q75 - q25)
    data_range: float = float(np.max(values) - np.min(values))
    if iqr <= 0.0 or data_range <= 0.0:
        return _calculate_sturges_bins(n)
    bin_width: float = 2.0 * iqr * (float(n) ** (-1.0 / 3.0))
    k_fd: int = int(np.ceil(data_range / bin_width))
    return max(1, min(k_fd, 250))


def _estimate_continuous_mode(values: np.ndarray) -> float:
    """Estima a moda de variável contínua pelo ponto médio da classe modal de Sturges."""
    k_bins: int = _calculate_sturges_bins(len(values))
    counts, edges = np.histogram(values, bins=k_bins)
    modal_idx: int = int(np.argmax(counts))
    return float(0.5 * (edges[modal_idx] + edges[modal_idx + 1]))


def _compute_series_metrics(values: np.ndarray, label: str) -> Dict[str, float | int | str]:
    """Calcula todas as métricas descritivas, de dispersão, formato e quantis para um vetor."""
    n: int = len(values)
    media: float = float(np.mean(values))
    mediana: float = float(np.median(values))
    moda_est: float = _estimate_continuous_mode(values)
    minimo: float = float(np.min(values))
    maximo: float = float(np.max(values))
    desvio_padrao: float = float(np.std(values, ddof=1))
    cv: float = desvio_padrao / abs(media) if media != 0.0 else np.nan

    p25, p50, p75, p90, p95, p99 = np.percentile(
        values, [25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
    )
    aiq: float = float(p75 - p25)
    cerca_sup_outlier: float = float(p75 + 1.5 * aiq)
    n_outliers_sup: int = int(np.sum(values > cerca_sup_outlier))

    assimetria: float = float(stats.skew(values, bias=False))
    curtose_excesso: float = float(stats.kurtosis(values, fisher=True, bias=False))
    curtose_bruta: float = float(stats.kurtosis(values, fisher=False, bias=False))

    return {
        "escopo": label,
        "n": n,
        "media": media,
        "mediana": mediana,
        "moda_classe_sturges": moda_est,
        "minimo": minimo,
        "maximo": maximo,
        "desvio_padrao": desvio_padrao,
        "coeficiente_variacao": cv,
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "aiq": aiq,
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "assimetria": assimetria,
        "curtose_excesso": curtose_excesso,
        "curtose_bruta": curtose_bruta,
        "limite_superior_15aiq": cerca_sup_outlier,
        "n_outliers_superiores": n_outliers_sup,
        "pct_outliers_superiores": 100.0 * n_outliers_sup / n,
    }


def analyze_q3_descriptive(
    df_perdas: pd.DataFrame,
    df_ataques: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Executa a Questão 3: estatísticas descritivas, quantis, Sturges e histogramas.

    Args:
        df_perdas: DataFrame `perdas.csv` contendo os incidentes materializados.
        df_ataques: DataFrame `ataques.csv` contendo todos os incidentes.
        tables_dir: Diretório `output/tables/` para exportação de tabelas CSV.
        figures_dir: Diretório `output/figures/` para exportação das figuras PNG.

    Returns:
        DataFrame consolidado com as estatísticas descritivas gerais e por tipo.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    perdas_val: np.ndarray = df_perdas["perda_rs"].to_numpy(dtype=float)
    ln_perdas_val: np.ndarray = np.log(perdas_val)

    linhas = [
        _compute_series_metrics(perdas_val, "Geral - perda_rs (R$)"),
        _compute_series_metrics(ln_perdas_val, "Geral - ln(perda_rs)"),
    ]

    # Estatísticas estratificadas por tipo de ataque (ordenadas pela mediana programada)
    for spec in TIPOS_ATAQUE:
        sub_vals: np.ndarray = df_perdas.loc[
            df_perdas["tipo"] == spec.nome, "perda_rs"
        ].to_numpy(dtype=float)
        if len(sub_vals) > 0:
            linhas.append(_compute_series_metrics(sub_vals, f"Tipo: {spec.nome} (R$)"))

    df_q3_stats = pd.DataFrame(linhas)
    out_stats_path: Path = tables_dir / "q3_estatisticas_descritivas.csv"
    df_q3_stats.to_csv(out_stats_path, index=False, float_format="%.4f")

    # -------------------------------------------------------------------------
    # Tabela complementar Q3-b: Distribuição de frequências de perda_rs (Sturges)
    # e distribuição de frequências por tipo de ataque
    # -------------------------------------------------------------------------
    k_sturges: int = _calculate_sturges_bins(len(perdas_val))
    k_fd: int = _calculate_freedman_diaconis_bins(perdas_val)

    counts_st, edges_st = np.histogram(perdas_val, bins=k_sturges)
    freq_rel_st = counts_st / len(perdas_val)
    df_freq_sturges = pd.DataFrame(
        {
            "classe_id": np.arange(1, k_sturges + 1, dtype=int),
            "limite_inferior_rs": edges_st[:-1],
            "limite_superior_rs": edges_st[1:],
            "ponto_medio_rs": 0.5 * (edges_st[:-1] + edges_st[1:]),
            "freq_absoluta": counts_st,
            "freq_relativa": freq_rel_st,
            "freq_acumulada_abs": np.cumsum(counts_st),
            "freq_acumulada_rel": np.cumsum(freq_rel_st),
        }
    )
    df_freq_sturges.to_csv(
        tables_dir / "q3_distribuicao_frequencias_sturges.csv",
        index=False,
        float_format="%.4f",
    )

    contagem_tipos = df_ataques["tipo"].value_counts()
    df_freq_tipos = pd.DataFrame(
        {
            "tipo": contagem_tipos.index,
            "freq_absoluta": contagem_tipos.to_numpy(dtype=int),
        }
    )
    df_freq_tipos["freq_relativa"] = df_freq_tipos["freq_absoluta"] / len(df_ataques)
    df_freq_tipos["freq_acumulada_abs"] = df_freq_tipos["freq_absoluta"].cumsum()
    df_freq_tipos["freq_acumulada_rel"] = df_freq_tipos["freq_relativa"].cumsum()
    df_freq_tipos.to_csv(
        tables_dir / "q3_distribuicao_frequencias_tipos.csv",
        index=False,
        float_format="%.4f",
    )

    # -------------------------------------------------------------------------
    # Figura Q3: Histograma de perdas (Sturges vs. Freedman-Diaconis e ln(perda))
    # -------------------------------------------------------------------------
    media_val: float = float(np.mean(perdas_val))
    p50_val, p90_val, p95_val, p99_val = np.percentile(
        perdas_val, [50.0, 90.0, 95.0, 99.0]
    )
    skew_raw: float = float(stats.skew(perdas_val, bias=False))
    kurt_raw: float = float(stats.kurtosis(perdas_val, fisher=True, bias=False))
    skew_ln: float = float(stats.skew(ln_perdas_val, bias=False))
    kurt_ln: float = float(stats.kurtosis(ln_perdas_val, fisher=True, bias=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), dpi=150)

    # Painel 1: Regra de Sturges em perda_rs
    axes[0].hist(
        perdas_val / 1e3,
        bins=k_sturges,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.78,
    )
    axes[0].axvline(
        media_val / 1e3,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Média: R$ {media_val/1e3:,.1f} mil",
    )
    axes[0].axvline(
        p50_val / 1e3,
        color="#2ca02c",
        linestyle="-",
        linewidth=1.8,
        label=f"Mediana (P50): R$ {p50_val/1e3:,.1f} mil",
    )
    axes[0].axvline(
        p95_val / 1e3,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.8,
        label=f"P95: R$ {p95_val/1e3:,.1f} mil",
    )
    axes[0].set_title(
        f"(a) Regra de Sturges ($k = {k_sturges}$ classes)\n"
        f"Assimetria = {skew_raw:.2f} | Curtose (exc.) = {kurt_raw:.2f}"
    )
    axes[0].set_xlabel("Perda por Incidente (R$ mil)")
    axes[0].set_ylabel("Frequência Absoluta")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Painel 2: Regra de Freedman-Diaconis (foco até P99 para visualização do núcleo e cauda)
    limite_vis_mil: float = float(p99_val / 1e3) * 1.15
    axes[1].hist(
        perdas_val / 1e3,
        bins=k_fd,
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.75,
    )
    axes[1].set_xlim(0.0, limite_vis_mil)
    axes[1].axvline(
        p50_val / 1e3,
        color="#2ca02c",
        linestyle="-",
        linewidth=1.8,
        label=f"P50: R$ {p50_val/1e3:,.1f} mil",
    )
    axes[1].axvline(
        p90_val / 1e3,
        color="#9467bd",
        linestyle="--",
        linewidth=1.6,
        label=f"P90: R$ {p90_val/1e3:,.1f} mil",
    )
    axes[1].axvline(
        p95_val / 1e3,
        color="#d62728",
        linestyle="-.",
        linewidth=1.6,
        label=f"P95: R$ {p95_val/1e3:,.1f} mil",
    )
    axes[1].axvline(
        p99_val / 1e3,
        color="black",
        linestyle=":",
        linewidth=1.6,
        label=f"P99: R$ {p99_val/1e3:,.1f} mil",
    )
    axes[1].set_title(
        f"(b) Regra de Freedman-Diaconis ($k = {k_fd}$ classes)\n"
        f"Zoom até 1,15 × P99 (Núcleo + Cauda)"
    )
    axes[1].set_xlabel("Perda por Incidente (R$ mil)")
    axes[1].set_ylabel("Frequência Absoluta")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Painel 3: Transformação logarítmica ln(perda_rs)
    axes[2].hist(
        ln_perdas_val,
        bins=k_sturges,
        density=True,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.72,
        label=f"Empírico ($k={k_sturges}$)",
    )
    x_ln = np.linspace(float(np.min(ln_perdas_val)), float(np.max(ln_perdas_val)), 300)
    mu_ln: float = float(np.mean(ln_perdas_val))
    sd_ln: float = float(np.std(ln_perdas_val, ddof=1))
    axes[2].plot(
        x_ln,
        stats.norm.pdf(x_ln, loc=mu_ln, scale=sd_ln),
        "r--",
        linewidth=2.0,
        label=f"Normal($\\hat{{\\mu}}={mu_ln:.2f}, \\hat{{\\sigma}}={sd_ln:.2f}$)",
    )
    axes[2].set_title(
        f"(c) Escala Logarítmica $\\ln(\\text{{perda\\_rs}})$\n"
        f"Assimetria = {skew_ln:.2f} | Curtose (exc.) = {kurt_ln:.2f}"
    )
    axes[2].set_xlabel("$\\ln(\\text{perda\\_rs})$")
    axes[2].set_ylabel("Densidade")
    axes[2].legend(loc="upper right", fontsize=8.5)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 3 e 4-c: Distribuição de Perdas Cibernéticas (Sturges vs. Freedman-Diaconis e Simetrização Logarítmica)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path: Path = figures_dir / "q3_histograma_perdas.png"
    fig.savefig(fig_path)
    plt.close(fig)

    logger.info(
        "Questão 3 concluída: tabela salva em %s e figura salva em %s.",
        out_stats_path,
        fig_path,
    )
    return df_q3_stats


def analyze_q4_visualizations(
    datasets: GeneratedDatasets,
    tables_dir: Path,
    figures_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Executa a Questão 4: box-plots por tipo, heatmap de correlação, scatterplots e VaR 95%.

    Args:
        datasets: Contêiner `GeneratedDatasets` com as 5 bases da Parte A.
        tables_dir: Diretório `output/tables/` para exportação de tabelas CSV.
        figures_dir: Diretório `output/figures/` para exportação das figuras PNG.

    Returns:
        Dicionário contendo as matrizes de correlação e métricas de VaR mensal.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_ataques = datasets.ataques
    df_perdas = datasets.perdas
    df_painel = datasets.painel_trimestral
    df_serie = datasets.serie_mensal

    ordem_tipos = [spec.nome for spec in TIPOS_ATAQUE]

    # -------------------------------------------------------------------------
    # 1. Boxplot de perda_rs por tipo (Escala Linear R$ e Escala Logarítmica)
    # -------------------------------------------------------------------------
    df_perdas_plot = df_perdas.copy()
    df_perdas_plot["perda_mil_rs"] = df_perdas_plot["perda_rs"] / 1e3
    df_perdas_plot["ln_perda_rs"] = np.log(df_perdas_plot["perda_rs"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150)

    sns.boxplot(
        data=df_perdas_plot,
        x="tipo",
        y="perda_mil_rs",
        order=ordem_tipos,
        hue="tipo",
        palette="Set2",
        legend=False,
        fliersize=3.5,
        ax=axes[0],
    )
    axes[0].set_title(
        "(a) Escala Linear (R$ mil) — Evidencia Outliers de Cauda ($> Q_3 + 1,5\\times\\text{AIQ}$)"
    )
    axes[0].set_xlabel("Tipo de Ataque")
    axes[0].set_ylabel("Perda Financeira (R$ mil)")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(
        data=df_perdas_plot,
        x="tipo",
        y="ln_perda_rs",
        order=ordem_tipos,
        hue="tipo",
        palette="Set2",
        legend=False,
        fliersize=3.5,
        ax=axes[1],
    )
    axes[1].set_title(
        "(b) Escala Logarítmica $\\ln(\\text{perda\\_rs})$ — Comparação de Medianas e AIQ"
    )
    axes[1].set_xlabel("Tipo de Ataque")
    axes[1].set_ylabel("$\\ln(\\text{perda\\_rs})$")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 4-b: Box-plot da Severidade de Perdas por Tipo de Ataque (ANDI)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    boxplot_path: Path = figures_dir / "q4_boxplot_perdas_por_tipo.png"
    fig.savefig(boxplot_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Heatmap de correlação para:
    #    servicos, patch, treino, soc, cvss, tempo_deteccao_h, materializou
    # -------------------------------------------------------------------------
    cols_corr_ataque = [
        "servicos",
        "patch",
        "treino",
        "soc",
        "cvss",
        "tempo_deteccao_h",
        "materializou",
    ]
    df_corr_ataques = df_ataques[cols_corr_ataque].corr(method="pearson")
    df_corr_ataques.to_csv(
        tables_dir / "q4_matriz_correlacao_ataques.csv", float_format="%.4f"
    )

    # Também calcula matriz de covariâncias e correlações de painel_trimestral.csv (Q4-d)
    cols_painel = ["n_ataques", "perda_total", "servicos", "patch", "treino", "soc"]
    df_corr_painel = df_painel[cols_painel].corr(method="pearson")
    df_cov_painel = df_painel[cols_painel].cov()
    df_corr_painel.to_csv(
        tables_dir / "q4_matriz_correlacao_painel.csv", float_format="%.4f"
    )
    df_cov_painel.to_csv(
        tables_dir / "q4_matriz_covariancia_painel.csv", float_format="%.4f"
    )

    # VaR histórico de 95% de serie_mensal.perda_total (Q4-e)
    var95_mensal: float = float(np.percentile(df_serie["perda_total"], 95.0))
    media_mensal: float = float(df_serie["perda_total"].mean())
    max_mensal: float = float(df_serie["perda_total"].max())
    df_var_hist = pd.DataFrame(
        [
            {
                "n_meses": len(df_serie),
                "media_perda_mensal_rs": media_mensal,
                "mediana_perda_mensal_rs": float(df_serie["perda_total"].median()),
                "var_historico_95_mensal_rs": var95_mensal,
                "maximo_perda_mensal_rs": max_mensal,
                "mes_maximo": str(
                    df_serie.loc[df_serie["perda_total"].idxmax(), "mes_ref"]
                ),
            }
        ]
    )
    df_var_hist.to_csv(
        tables_dir / "q4_var_historico_mensal.csv", index=False, float_format="%.2f"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150)

    sns.heatmap(
        df_corr_ataques,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axes[0],
    )
    axes[0].set_title(
        "(a) Correlações no Nível do Incidente (`ataques.csv`)\n"
        "[servicos, patch, treino, soc, cvss, tempo_deteccao_h, materializou]"
    )

    sns.heatmap(
        df_corr_painel,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axes[1],
    )
    axes[1].set_title(
        "(b) Correlações no Painel Trimestral (`painel_trimestral.csv`)\n"
        "[n_ataques, perda_total, servicos, patch, treino, soc]"
    )

    fig.suptitle(
        "Questão 4: Mapas de Calor de Correlação de Pearson — Nível de Ataque e Painel Trimestral",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    heatmap_path: Path = figures_dir / "q4_heatmap_correlacao.png"
    fig.savefig(heatmap_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Scatterplots:
    #    - patch x materializou
    #    - treino x materializou
    #    - cvss x perda_rs
    # -------------------------------------------------------------------------
    rng_vis = np.random.default_rng(SEED + 400)
    jitter_y = rng_vis.uniform(-0.04, 0.04, size=len(df_ataques))

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), dpi=150)

    # Painel (a): patch x materializou (com taxa empírica por unidade e tendência)
    df_unit_mat = (
        df_ataques.groupby(["unidade", "patch", "treino", "soc"], as_index=False)
        .agg(taxa_mat=("materializou", "mean"), n_atq=("id_ataque", "count"))
    )

    axes[0].scatter(
        df_ataques["patch"],
        df_ataques["materializou"] + jitter_y,
        alpha=0.08,
        s=10,
        color="#7f7f7f",
        label="Ataques (0/1 c/ jitter)",
    )
    axes[0].scatter(
        df_unit_mat["patch"],
        df_unit_mat["taxa_mat"],
        c=df_unit_mat["soc"],
        cmap="bwr",
        s=65,
        edgecolor="black",
        zorder=4,
        label="Taxa de Mat. por Unidade",
    )
    sns.regplot(
        data=df_ataques,
        x="patch",
        y="materializou",
        scatter=False,
        ci=95,
        color="#1f77b4",
        ax=axes[0],
        line_kws={"linewidth": 2.0, "label": "Tendência observada na amostra"},
    )
    axes[0].set_title("(a) Scatterplot: `patch` × `materializou`")
    axes[0].set_xlabel("Cobertura de Patches da Unidade (%)")
    axes[0].set_ylabel("Materialização (0 = Não, 1 = Sim / Taxa)")
    axes[0].set_ylim(-0.10, 1.10)
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Painel (b): treino x materializou
    axes[1].scatter(
        df_ataques["treino"],
        df_ataques["materializou"] + jitter_y,
        alpha=0.08,
        s=10,
        color="#7f7f7f",
        label="Ataques (0/1 c/ jitter)",
    )
    axes[1].scatter(
        df_unit_mat["treino"],
        df_unit_mat["taxa_mat"],
        c=df_unit_mat["soc"],
        cmap="bwr",
        s=65,
        edgecolor="black",
        zorder=4,
        label="Taxa de Mat. por Unidade",
    )
    sns.regplot(
        data=df_ataques,
        x="treino",
        y="materializou",
        scatter=False,
        ci=95,
        color="#2ca02c",
        ax=axes[1],
        line_kws={"linewidth": 2.0, "label": "Tendência observada na amostra"},
    )
    axes[1].set_title("(b) Scatterplot: `treino` × `materializou`")
    axes[1].set_xlabel("Treinamento Anual de Conscientização (horas)")
    axes[1].set_ylabel("Materialização (0 = Não, 1 = Sim / Taxa)")
    axes[1].set_ylim(-0.10, 1.10)
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Painel (c): cvss x perda_rs (cruzando ataques materializados com perdas)
    df_cvss_perda = df_perdas.merge(
        df_ataques[["id_ataque", "cvss", "soc"]], on="id_ataque", how="left"
    )
    sns.scatterplot(
        data=df_cvss_perda,
        x="cvss",
        y="perda_rs",
        hue="tipo",
        hue_order=ordem_tipos,
        palette="Set2",
        alpha=0.72,
        s=32,
        ax=axes[2],
    )
    axes[2].set_yscale("log")
    axes[2].set_title("(c) Scatterplot: `cvss` × `perda_rs` (Escala Log)")
    axes[2].set_xlabel("Gravidade Técnica (CVSS: 0 a 10)")
    axes[2].set_ylabel("Perda Financeira `perda_rs` (R$ - Escala Log)")
    axes[2].legend(loc="upper left", fontsize=7.5, title="Tipo de Ataque")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Questão 4: Diagramas de Dispersão (Patch × Materializou, Treino × Materializou e CVSS × Perda)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    scatter_path: Path = figures_dir / "q4_scatterplots.png"
    fig.savefig(scatter_path)
    plt.close(fig)

    logger.info(
        "Questão 4 concluída: figuras salvas em %s, %s e %s.",
        boxplot_path,
        heatmap_path,
        scatter_path,
    )
    return {
        "correlacao_ataques": df_corr_ataques,
        "correlacao_painel": df_corr_painel,
        "covariancia_painel": df_cov_painel,
        "var_historico_mensal": df_var_hist,
    }