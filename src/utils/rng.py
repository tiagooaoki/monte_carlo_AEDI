"""Utilitário de gerenciamento do Gerador de Números Pseudoaleatórios (RNG).

Encapsula a criação de instâncias de `numpy.random.Generator` (PCG64) a partir da
semente oficial do Grupo 03 (SEED = 202603) e bloqueia o uso acidental da API
global legada `np.random`.
"""

from typing import Optional
import numpy as np
from numpy.random import Generator

from src.config import SEED


def _blocked_legacy_rng(*args: object, **kwargs: object) -> None:
    """Lança exceção caso algum módulo tente invocar `np.random.seed` global."""
    raise RuntimeError(
        "Proibido usar o estado global legado `np.random.seed()`. "
        "Utilize `get_rng()` de `src.utils.rng` para obter um `np.random.Generator` isolado."
    )


def disable_legacy_global_rng() -> None:
    """Bloqueia chamadas diretas a `np.random.seed` para forçar o uso de `default_rng`."""
    np.random.seed = _blocked_legacy_rng  # type: ignore[assignment]


def get_rng(seed: Optional[int] = None) -> Generator:
    """Cria e retorna uma instância isolada do gerador `numpy.random.Generator`.

    Args:
        seed: Semente inteira opcional. Se `None`, utiliza a constante oficial
            `SEED` (202603) definida em `src.config`.

    Returns:
        Instância configurada de `numpy.random.Generator` (BitGenerator PCG64).

    Raises:
        TypeError: Se a semente fornecida não for um número inteiro.
    """
    effective_seed: int = SEED if seed is None else seed
    if not isinstance(effective_seed, int) or isinstance(effective_seed, bool):
        raise TypeError(f"A semente deve ser um inteiro válido. Recebido: {type(effective_seed)}")

    return np.random.default_rng(effective_seed)


def validate_rng_determinism(seed: int = SEED, sample_size: int = 10) -> bool:
    """Verifica se duas instâncias criadas com a mesma semente geram sequências idênticas.

    Args:
        seed: Semente a ser validada.
        sample_size: Quantidade de números uniformes a comparar.

    Returns:
        `True` se os vetores sorteados forem estritamente iguais.

    Raises:
        AssertionError: Se houver divergência entre os sorteios ou tipo inválido.
    """
    rng_a: Generator = get_rng(seed)
    rng_b: Generator = get_rng(seed)

    if not isinstance(rng_a, Generator):
        raise AssertionError("O objeto retornado não é um `numpy.random.Generator`.")

    draw_a: np.ndarray = rng_a.uniform(0.0, 1.0, size=sample_size)
    draw_b: np.ndarray = rng_b.uniform(0.0, 1.0, size=sample_size)

    if not np.array_equal(draw_a, draw_b):
        raise AssertionError("Falha de determinismo: a mesma semente produziu vetores distintos.")

    return True