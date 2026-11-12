"""Módulo de transformações inversas F⁻¹(U) e validação estatística (Questão 1-b).

Implementa matematicamente a geração de variáveis aleatórias Exponenciais e
Triangulares a partir de sorteios Uniformes U ~ U(0, 1) utilizando exclusivamente
o gerador encapsulado do projeto, e compara os resultados contra as implementações
nativas do NumPy via momentos (média e variância), teste de Kolmogorov-Smirnov (KS)
e histogramas sobrepostos à densidade teórica.
"""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(name="parte_a.inverse_transform")


def inverse_exponential(
    rng: Generator,
    theta: float,
    size: int,
) -> np.ndarray:
    """Gera amostras da distribuição Exponencial(theta) via transformação inversa.

    Dedução matemática:
        F(x) = 1 - exp(-x / theta), para x >= 0 e theta > 0.
        Igualando u = F(x) => 1 - u = exp(-x / theta) => x = -theta * ln(1 - u).

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        theta: Parâmetro de escala (média teórica E[X] = theta > 0).
        size: Quantidade de observações a sortear.

    Returns:
        Vetor NumPy unidimensional com os valores sorteados.

    Raises:
        ValueError: Se `theta <= 0` ou `size <= 0`.
    """
    if theta <= 0.0:
        raise ValueError(f"O parâmetro theta deve ser estritamente positivo. Recebido: {theta}")
    if size <= 0:
        raise ValueError(f"O tamanho da amostra deve ser positivo. Recebido: {size}")

    u: np.ndarray = rng.uniform(0.0, 1.0, size=size)
    return -theta * np.log(1.0 - u)


def inverse_triangular(
    rng: Generator,
    left: float,
    mode: float,
    right: float,
    size: int,
) -> np.ndarray:
    """Gera amostras da distribuição Triangular(left, mode, right) via transformada inversa.

    Dedução matemática nos dois ramos (a = left, c = mode, b = right):
        Ponto de inflexão acumulado na moda: u_c = F(c) = (c - a) / (b - a).
        Ramo 1 (x <= c <=> u <= u_c):
            F(x) = (x - a)^2 / [(b - a)(c - a)]
            F⁻¹(u) = a + sqrt(u * (b - a) * (c - a))
        Ramo 2 (x > c <=> u > u_c):
            F(x) = 1 - (b - x)^2 / [(b - a)(b - c)]
            F⁻¹(u) = b - sqrt((1 - u) * (b - a) * (b - c))

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        left: Limite inferior (a) da distribuição.
        mode: Moda (c) da distribuição, com left <= mode <= right.
        right: Limite superior (b) da distribuição, com right > left.
        size: Quantidade de observações a sortear.

    Returns:
        Vetor NumPy unidimensional com os valores sorteados.

    Raises:
        ValueError: Se os parâmetros violarem a condição left <= mode <= right ou left >= right.
    """
    if not (left <= mode <= right) or (left >= right):
        raise ValueError(
            f"Parâmetros inválidos para Triangular: left={left}, mode={mode}, right={right}."
        )
    if size <= 0:
        raise ValueError(f"O tamanho da amostra deve ser positivo. Recebido: {size}")

    u: np.ndarray = rng.uniform(0.0, 1.0, size=size)
    span: float = right - left
    u_critico: float = (mode - left) / span

    ramo_esquerdo: np.ndarray = left + np.sqrt(
        np.maximum(0.0, u * span * (mode - left))
    )
    ramo_direito: np.ndarray = right - np.sqrt(
        np.maximum(0.0, (1.0 - u) * span * (right - mode))
    )

    return np.where(u <= u_critico, ramo_esquerdo, ramo_direito)


def compare_exponential_generators(
    rng: Generator,
    theta: float = 12.0,
    size: int = 100_000,
    validation_dir: Path | None = None,
) -> Dict[str, float | str]:
    """Compara `inverse_exponential` com `rng.exponential` e salva gráfico de validação.

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        theta: Média teórica da distribuição Exponencial avaliada.
        size: Tamanho amostral para o confronto estatístico.
        validation_dir: Diretório `output/validation/` onde a figura será salva.

    Returns:
        Dicionário contendo médias, variâncias e estatísticas/p-valores do teste KS.
    """
    sample_inv: np.ndarray = inverse_exponential(rng=rng, theta=theta, size=size)
    sample_lib: np.ndarray = rng.exponential(scale=theta, size=size)

    media_teorica: float = theta
    var_teorica: float = theta**2

    ks_2samp_stat, ks_2samp_pval = stats.ks_2samp(sample_inv, sample_lib)
    ks_1samp_stat, ks_1samp_pval = stats.ks_1samp(
        sample_inv, stats.expon(scale=theta).cdf
    )

    if validation_dir is not None:
        validation_dir.mkdir(parents=True, exist_ok=True)
        fig_path: Path = validation_dir / "comparacao_exponencial.png"

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        x_grid: np.ndarray = np.linspace(0.0, np.percentile(sample_inv, 99.5), 400)
        pdf_teorica: np.ndarray = stats.expon.pdf(x_grid, scale=theta)

        ax.hist(
            sample_inv,
            bins=70,
            density=True,
            alpha=0.50,
            color="#1f77b4",
            label=f"Transformada Inversa F⁻¹(U) (Média={sample_inv.mean():.3f})",
        )
        ax.hist(
            sample_lib,
            bins=70,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#ff7f0e",
            label=f"NumPy rng.exponential (Média={sample_lib.mean():.3f})",
        )
        ax.plot(
            x_grid,
            pdf_teorica,
            "r--",
            linewidth=2.0,
            label=f"FDP Teórica Exponencial(θ={theta:.1f})",
        )
        ax.set_title(
            f"Validação Q1-b: Exponencial(θ={theta:.1f}h) — Transformada Inversa vs. Biblioteca\n"
            f"Teste KS (2 amostras): D = {ks_2samp_stat:.5f}, p-valor = {ks_2samp_pval:.4f}"
        )
        ax.set_xlabel("Tempo de Detecção (horas)")
        ax.set_ylabel("Densidade")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info("Gráfico de validação da Exponencial salvo em: %s", fig_path)

    return {
        "distribuicao": f"Exponencial(theta={theta})",
        "n_amostras": size,
        "media_teorica": media_teorica,
        "media_inversa": float(np.mean(sample_inv)),
        "media_numpy": float(np.mean(sample_lib)),
        "variancia_teorica": var_teorica,
        "variancia_inversa": float(np.var(sample_inv, ddof=1)),
        "variancia_numpy": float(np.var(sample_lib, ddof=1)),
        "ks_1samp_stat_teorica": float(ks_1samp_stat),
        "ks_1samp_pvalor_teorica": float(ks_1samp_pval),
        "ks_2samp_stat_vs_numpy": float(ks_2samp_stat),
        "ks_2samp_pvalor_vs_numpy": float(ks_2samp_pval),
    }


def compare_triangular_generators(
    rng: Generator,
    left: float = 60.0,
    mode: float = 80.0,
    right: float = 95.0,
    size: int = 100_000,
    validation_dir: Path | None = None,
) -> Dict[str, float | str]:
    """Compara `inverse_triangular` com `rng.triangular` e salva gráfico de validação.

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        left: Limite inferior (a) da Triangular.
        mode: Moda (c) da Triangular.
        right: Limite superior (b) da Triangular.
        size: Tamanho amostral para o confronto estatístico.
        validation_dir: Diretório `output/validation/` onde a figura será salva.

    Returns:
        Dicionário contendo médias, variâncias e estatísticas/p-valores do teste KS.
    """
    sample_inv: np.ndarray = inverse_triangular(
        rng=rng, left=left, mode=mode, right=right, size=size
    )
    sample_lib: np.ndarray = rng.triangular(
        left=left, mode=mode, right=right, size=size
    )

    media_teorica: float = (left + mode + right) / 3.0
    var_teorica: float = (
        left**2 + mode**2 + right**2 - left * mode - left * right - mode * right
    ) / 18.0

    c_scipy: float = (mode - left) / (right - left)
    dist_teorica = stats.triang(c=c_scipy, loc=left, scale=right - left)

    ks_2samp_stat, ks_2samp_pval = stats.ks_2samp(sample_inv, sample_lib)
    ks_1samp_stat, ks_1samp_pval = stats.ks_1samp(sample_inv, dist_teorica.cdf)

    if validation_dir is not None:
        validation_dir.mkdir(parents=True, exist_ok=True)
        fig_path: Path = validation_dir / "comparacao_triangular.png"

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        x_grid: np.ndarray = np.linspace(left, right, 400)
        pdf_teorica: np.ndarray = dist_teorica.pdf(x_grid)

        ax.hist(
            sample_inv,
            bins=70,
            density=True,
            alpha=0.50,
            color="#2ca02c",
            label=f"Transformada Inversa F⁻¹(U) (Média={sample_inv.mean():.3f})",
        )
        ax.hist(
            sample_lib,
            bins=70,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#d62728",
            label=f"NumPy rng.triangular (Média={sample_lib.mean():.3f})",
        )
        ax.plot(
            x_grid,
            pdf_teorica,
            "k--",
            linewidth=2.0,
            label=f"FDP Teórica Triangular({left:.0f}; {mode:.0f}; {right:.0f})",
        )
        ax.set_title(
            f"Validação Q1-b: Triangular({left:.0f}; {mode:.0f}; {right:.0f}) — Transformada Inversa vs. Biblioteca\n"
            f"Teste KS (2 amostras): D = {ks_2samp_stat:.5f}, p-valor = {ks_2samp_pval:.4f}"
        )
        ax.set_xlabel("Valor da Variável (Cobertura de Patches %)")
        ax.set_ylabel("Densidade")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info("Gráfico de validação da Triangular salvo em: %s", fig_path)

    return {
        "distribuicao": f"Triangular({left}; {mode}; {right})",
        "n_amostras": size,
        "media_teorica": media_teorica,
        "media_inversa": float(np.mean(sample_inv)),
        "media_numpy": float(np.mean(sample_lib)),
        "variancia_teorica": var_teorica,
        "variancia_inversa": float(np.var(sample_inv, ddof=1)),
        "variancia_numpy": float(np.var(sample_lib, ddof=1)),
        "ks_1samp_stat_teorica": float(ks_1samp_stat),
        "ks_1samp_pvalor_teorica": float(ks_1samp_pval),
        "ks_2samp_stat_vs_numpy": float(ks_2samp_stat),
        "ks_2samp_pvalor_vs_numpy": float(ks_2samp_pval),
    }


def run_inverse_transform_validation(
    rng: Generator,
    validation_dir: Path,
    tables_dir: Path,
    sample_size: int = 100_000,
) -> pd.DataFrame:
    """Executa a bateria completa de validação da Questão 1-b e exporta tabela resumo.

    Args:
        rng: Instância isolada de `numpy.random.Generator`.
        validation_dir: Caminho para `output/validation/`.
        tables_dir: Caminho para `output/tables/`.
        sample_size: Quantidade de réplicas de Monte Carlo para os testes KS.

    Returns:
        DataFrame contendo as métricas comparativas das duas distribuições.
    """
    validation_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    res_exp_sem_soc = compare_exponential_generators(
        rng=rng, theta=12.0, size=sample_size, validation_dir=validation_dir
    )
    res_exp_com_soc = compare_exponential_generators(
        rng=rng, theta=4.0, size=sample_size, validation_dir=None
    )
    res_tri = compare_triangular_generators(
        rng=rng, left=60.0, mode=80.0, right=95.0, size=sample_size, validation_dir=validation_dir
    )

    df_resumo = pd.DataFrame([res_exp_com_soc, res_exp_sem_soc, res_tri])
    out_csv: Path = tables_dir / "validacao_transformada_inversa.csv"
    df_resumo.to_csv(out_csv, index=False, float_format="%.6f")
    logger.info("Tabela de validação da transformada inversa exportada em: %s", out_csv)

    return df_resumo