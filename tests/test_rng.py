"""Testes automatizados de reprodutibilidade e isolamento do gerador aleatório (RNG)."""

import numpy as np
import pytest
from numpy.random import Generator

from src.config import SEED
from src.utils.rng import (
    disable_legacy_global_rng,
    get_rng,
    validate_rng_determinism,
)


def test_official_group_seed_value() -> None:
    """Verifica se a semente oficial do Grupo 03 está fixada em 202603."""
    assert SEED == 202603


def test_get_rng_returns_numpy_generator() -> None:
    """Verifica se `get_rng` retorna uma instância moderna de `np.random.Generator`."""
    rng: Generator = get_rng()
    assert isinstance(rng, Generator)


def test_same_seed_produces_identical_numbers() -> None:
    """Valida que duas instâncias com a mesma semente geram exatamente os mesmos valores."""
    rng_1: Generator = get_rng(SEED)
    rng_2: Generator = get_rng(SEED)

    # Testa sorteios uniformes, normais e inteiros em sequência
    uniform_1: np.ndarray = rng_1.uniform(0.0, 1.0, size=1_000)
    uniform_2: np.ndarray = rng_2.uniform(0.0, 1.0, size=1_000)
    np.testing.assert_array_equal(uniform_1, uniform_2)

    normal_1: np.ndarray = rng_1.normal(loc=0.0, scale=0.9, size=500)
    normal_2: np.ndarray = rng_2.normal(loc=0.0, scale=0.9, size=500)
    np.testing.assert_array_equal(normal_1, normal_2)

    integers_1: np.ndarray = rng_1.integers(10, 31, size=24)
    integers_2: np.ndarray = rng_2.integers(10, 31, size=24)
    np.testing.assert_array_equal(integers_1, integers_2)


def test_different_seeds_produce_different_numbers() -> None:
    """Valida que sementes distintas produzem sequências numéricas diferentes."""
    rng_grupo_03: Generator = get_rng(202603)
    rng_grupo_04: Generator = get_rng(202604)

    draw_g3: np.ndarray = rng_grupo_03.uniform(0.0, 1.0, size=100)
    draw_g4: np.ndarray = rng_grupo_04.uniform(0.0, 1.0, size=100)

    assert not np.array_equal(draw_g3, draw_g4)


def test_validate_rng_determinism_helper() -> None:
    """Verifica a função auxiliar de validação invocada pelo `main.py`."""
    assert validate_rng_determinism(seed=SEED) is True


def test_invalid_seed_raises_type_error() -> None:
    """Verifica se passar tipos inválidos como semente dispara `TypeError`."""
    with pytest.raises(TypeError):
        get_rng("202603")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        get_rng(True)  # type: ignore[arg-type]


def test_legacy_global_rng_is_blocked() -> None:
    """Confirma que `disable_legacy_global_rng` bloqueia chamadas a `np.random.seed`."""
    disable_legacy_global_rng()
    with pytest.raises(RuntimeError, match="Proibido usar o estado global legado"):
        np.random.seed(202603)

def test_ataques_contains_unit_profile_columns() -> None:

    rng = get_rng(SEED)

    unidades = generate_etapa_a1_unidades(rng=rng)

    grade = generate_etapa_a2_contagens(
        rng=rng,
        df_unidades=unidades,
    )

    ataques = generate_etapas_a3_a4_ataques(
        rng=rng,
        df_grade=grade,
    )

    assert "soc" in ataques.columns
    assert "treino" in ataques.columns
    assert "patch" in ataques.columns
    assert "servicos" in ataques.columns