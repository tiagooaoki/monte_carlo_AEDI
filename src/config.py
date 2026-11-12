"""Módulo de configuração central do projeto AEDI - Estudo de Caso ANDI.

Centraliza a semente oficial do Grupo 03, os caminhos de diretórios e todos os
parâmetros verdadeiros do modelo gerador (Etapas A1 a A5) e das análises (Partes B e C).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Tuple

# =============================================================================
# IDENTIFICAÇÃO DO GRUPO E SEMENTE OFICIAL (REPRODUTIBILIDADE)
# =============================================================================
GROUP_NUMBER: Final[int] = 3
GROUP_NAME: Final[str] = "Grupo 03"
SEED: Final[int] = 202600 + GROUP_NUMBER  # 202603

# Nível de significância estatística padrão (Regra de Conduta #3)
DEFAULT_ALPHA: Final[float] = 0.05

# =============================================================================
# CAMINHOS DE DIRETÓRIOS E ARQUIVOS (PATHLIB)
# =============================================================================
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = BASE_DIR / "data"
OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
FIGURES_DIR: Final[Path] = OUTPUT_DIR / "figures"
TABLES_DIR: Final[Path] = OUTPUT_DIR / "tables"
MODELS_DIR: Final[Path] = OUTPUT_DIR / "models"
DOCS_DIR: Final[Path] = BASE_DIR / "docs"
LOGS_DIR: Final[Path] = BASE_DIR / "logs"

# Caminhos exatos das 5 bases obrigatórias a entregar
UNIDADES_CSV: Final[Path] = DATA_DIR / "unidades.csv"
ATAQUES_CSV: Final[Path] = DATA_DIR / "ataques.csv"
PERDAS_CSV: Final[Path] = DATA_DIR / "perdas.csv"
PAINEL_TRIMESTRAL_CSV: Final[Path] = DATA_DIR / "painel_trimestral.csv"
SERIE_MENSAL_CSV: Final[Path] = DATA_DIR / "serie_mensal.csv"

# =============================================================================
# HORIZONTE TEMPORAL DO ESTUDO (PARTE A)
# =============================================================================
N_UNIDADES: Final[int] = 24
N_MESES: Final[int] = 36
N_TRIMESTRES: Final[int] = 12
ANO_INICIAL: Final[int] = 2023
ANO_FINAL: Final[int] = 2025


# =============================================================================
# ETAPA A1 — PERFIL DAS 24 UNIDADES (FIXO NOS 36 MESES)
# =============================================================================
@dataclass(frozen=True)
class EtapaA1Config:
    """Parâmetros das distribuições de entrada do perfil das 24 unidades."""

    # servicos: Uniforme discreta em {10, 11, ..., 30}
    servicos_min: int = 10
    servicos_max: int = 30

    # patch: Cobertura de patches (%) ~ Triangular(mín=60, moda=80, máx=95)
    patch_min: float = 60.0
    patch_mode: float = 80.0
    patch_max: float = 95.0

    # treino: Horas de treinamento por ano ~ Uniforme contínua(0, 12)
    treino_min: float = 0.0
    treino_max: float = 12.0

    # soc: Possui SOC dedicado (1 = sim) ~ Bernoulli(p = 0,40)
    soc_prob: float = 0.40


# =============================================================================
# ETAPA A2 — FREQUÊNCIA: NÚMERO DE ATAQUES POR UNIDADE E MÊS
# =============================================================================
@dataclass(frozen=True)
class EtapaA2Config:
    """Parâmetros da regressão de frequência trimestral, sazonalidade e campanha."""

    # Coeficientes de lambda_trim_{u,q} = 8 + 0.50*servicos - 0.08*patch - 0.20*treino - 1.40*soc + 0.15*q
    beta_0: float = 8.0
    beta_servicos: float = 0.50
    beta_patch: float = -0.08
    beta_treino: float = -0.20
    beta_soc: float = -1.40
    beta_trimestre: float = 0.15

    # Campanha coordenada de ataques em t = 28 (abril de 2025)
    mes_campanha_t: int = 28
    fator_campanha_c28: float = 1.80
    fator_campanha_padrao: float = 1.00


# Fatores sazonais mensais S_m (m = 1 a 12, média = 1,00)
FATORES_SAZONAIS: Final[Dict[int, float]] = {
    1: 0.90,   # Jan
    2: 0.85,   # Fev
    3: 1.10,   # Mar
    4: 1.10,   # Abr
    5: 1.00,   # Mai
    6: 0.95,   # Jun
    7: 0.90,   # Jul
    8: 0.95,   # Ago
    9: 1.00,   # Set
    10: 1.05,  # Out
    11: 1.15,  # Nov
    12: 1.05,  # Dez
}


# =============================================================================
# ETAPA A3 — ATRIBUTOS DE CADA ATAQUE E TIPOS DE ATAQUE
# =============================================================================
@dataclass(frozen=True)
class AttackTypeSpec:
    """Especificação de probabilidade, CVSS Triangular e mediana de perda por tipo."""

    nome: str
    probabilidade: float
    cvss_min: float
    cvss_mode: float
    cvss_max: float
    mediana_perda_rs: float


TIPOS_ATAQUE: Final[Tuple[AttackTypeSpec, ...]] = (
    AttackTypeSpec(
        nome="Phishing",
        probabilidade=0.35,
        cvss_min=2.0,
        cvss_mode=5.0,
        cvss_max=8.0,
        mediana_perda_rs=15_000.0,
    ),
    AttackTypeSpec(
        nome="Força bruta",
        probabilidade=0.20,
        cvss_min=2.0,
        cvss_mode=4.0,
        cvss_max=7.0,
        mediana_perda_rs=10_000.0,
    ),
    AttackTypeSpec(
        nome="Malware",
        probabilidade=0.15,
        cvss_min=4.0,
        cvss_mode=6.0,
        cvss_max=9.0,
        mediana_perda_rs=30_000.0,
    ),
    AttackTypeSpec(
        nome="Exploração de vulnerabilidade",
        probabilidade=0.15,
        cvss_min=5.0,
        cvss_mode=7.5,
        cvss_max=10.0,
        mediana_perda_rs=45_000.0,
    ),
    AttackTypeSpec(
        nome="DDoS",
        probabilidade=0.10,
        cvss_min=3.0,
        cvss_mode=5.0,
        cvss_max=8.0,
        mediana_perda_rs=25_000.0,
    ),
    AttackTypeSpec(
        nome="Ransomware",
        probabilidade=0.05,
        cvss_min=6.0,
        cvss_mode=8.5,
        cvss_max=10.0,
        mediana_perda_rs=150_000.0,
    ),
)


@dataclass(frozen=True)
class EtapaA3Config:
    """Parâmetros de horário e tempo de detecção de cada ataque."""

    # fora_expediente ~ Bernoulli(p = 0,35)
    fora_expediente_prob: float = 0.35

    # tempo_deteccao_h ~ Exponencial(média = theta)
    theta_com_soc_h: float = 4.0
    theta_sem_soc_h: float = 12.0

    # Multiplicador do tempo sorteado quando fora_expediente == 1
    multiplicador_fora_expediente: float = 1.50

    # Casas decimais de arredondamento do CVSS
    cvss_decimais: int = 1


# =============================================================================
# ETAPA A4 — MATERIALIZAÇÃO EM PERDA (MODELO LOGÍSTICO)
# =============================================================================
@dataclass(frozen=True)
class EtapaA4Config:
    """Coeficientes verdadeiros do logit(p_i) de materialização do ataque."""

    # logit(p_i) = -2.8 + 0.45*cvss + 0.06*tempo - 0.80*soc - 0.08*treino + 0.50*fora
    beta_0: float = -2.80
    beta_cvss: float = 0.45
    beta_tempo: float = 0.06
    beta_soc: float = -0.80
    beta_treino: float = -0.08
    beta_fora_expediente: float = 0.50


# =============================================================================
# ETAPA A5 — SEVERIDADE (MODELO LOGNORMAL CONDICIONAL)
# =============================================================================
@dataclass(frozen=True)
class EtapaA5Config:
    """Parâmetros verdadeiros da equação de severidade ln(perda_i)."""

    # ln(perda_i) = ln(mediana_tipo) + 0.02*tempo_i + erro_i, erro_i ~ N(0, 0.9^2)
    beta_tempo: float = 0.02
    sigma_erro: float = 0.90


# =============================================================================
# PARÂMETROS DE CLASSIFICAÇÃO E DECISÃO EXECUTIVA (QUESTÕES 13 E 14)
# =============================================================================
@dataclass(frozen=True)
class ParteCConfig:
    """Parâmetros de simulação e avaliação financeira para 2026 (Q13 e Q14)."""

    # Q13-c/d/e: Parâmetros de classificação e custo de triagem
    arvore_max_depth: int = 3
    limiar_padrao: float = 0.50
    custo_falso_positivo_rs: float = 3_000.0

    # Q14: Horizonte 2026 (trimestres 13 a 16) e réplicas da 2ª SMC
    trimestres_2026: Tuple[int, int, int, int] = (13, 14, 15, 16)
    n_replicas_smc: int = 100_000
    grid_convergencia: Tuple[int, ...] = (1_000, 4_000, 16_000, 64_000, 100_000)
    limiar_perda_catastrofica_rs: float = 30_000_000.0

    # Q14-d/e: Pacote de controles propostos para todas as 24 unidades
    custo_anual_controles_rs: float = 8_000_000.0
    controle_patch: float = 95.0
    controle_treino: float = 12.0
    controle_soc: int = 1

    # Q14-f: Parâmetros de sensibilidade
    variacao_sigma_lognormal: float = 0.20
    variacao_frequencia: float = 0.10

    # Q14-g: Anos simulados no modelo verdadeiro para medir erro de modelo
    n_anos_erro_modelo: int = 10_000


# Instâncias globais imutáveis prontas para importação nos módulos
ETAPA_A1: Final[EtapaA1Config] = EtapaA1Config()
ETAPA_A2: Final[EtapaA2Config] = EtapaA2Config()
ETAPA_A3: Final[EtapaA3Config] = EtapaA3Config()
ETAPA_A4: Final[EtapaA4Config] = EtapaA4Config()
ETAPA_A5: Final[EtapaA5Config] = EtapaA5Config()
PARTE_C: Final[ParteCConfig] = ParteCConfig()