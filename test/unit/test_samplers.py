import pytest
import numpy as np

from bneqpri.types import (
    NPDistribution,
    MVNDistribution,
    LognormalDistribution,
    BetaDistribution,
    ParetoDistribution,
    UniformDistribution,
)
from bneqpri.samplers import (
    ABVSampler,
    ABVIndependentSampler,
    AVIndependentSampler,
    LognormalABNormalVSampler,
    LognormalABNormalWXSampler,
)


@pytest.mark.parametrize("I", (10, 100, 1000))
def test_ABVSampler(I: int) -> None:  # noqa: E741
    S = ABVSampler()
    with pytest.raises(NotImplementedError):
        S(I)


@pytest.mark.parametrize("I", (10, 100, 1000))
@pytest.mark.parametrize("J", (2, 5))
def test_LogNormalAVIndependentSampler(I: int, J: int) -> None:  # noqa: E741

    a_dist = LognormalDistribution(1.0, 0.5)
    V_dist = MVNDistribution(np.zeros(J), np.eye(J))

    S = AVIndependentSampler(a_dist, V_dist)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b is None
    assert V.shape == (I, J)


@pytest.mark.parametrize("I", (10, 100, 1000))
@pytest.mark.parametrize("J", (2, 5))
def test_LognormalABNormalVSampler(I: int, J: int) -> None:  # noqa: E741
    mean_V = np.random.randn(J)
    cov_V = np.random.randn(J, J)
    cov_V = cov_V.T @ cov_V  # make SPD
    S = LognormalABNormalVSampler(1.0, 1.0, 1.0, 1.0, mean_V, cov_V)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b is not None and b.shape == (I,)
    assert V.shape == (I, J)


@pytest.mark.parametrize("I", (10, 100, 1000))
@pytest.mark.parametrize("J", (2, 5))
@pytest.mark.parametrize("K", (2,))
def test_LognormalABNormalWXSampler(I: int, J: int, K: int) -> None:  # noqa: E741
    mean_W = np.random.randn(K)
    cov_W = np.random.randn(K, K)
    cov_W = cov_W.T @ cov_W  # make SPD
    data_X = np.random.randn(K, J)
    S = LognormalABNormalWXSampler(1.0, 1.0, 1.0, 1.0, mean_W, cov_W, data_X)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b is not None and b.shape == (I,)
    assert V.shape == (I, J)


@pytest.mark.parametrize("a_dist", (BetaDistribution(1.0, 0.5), LognormalDistribution(1.0, 0.5)))
@pytest.mark.parametrize("b_dist", (ParetoDistribution(1.0), UniformDistribution(2.0, 10.0)))
@pytest.mark.parametrize(
    "V_dist", (MVNDistribution(np.array([2.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])),)
)
@pytest.mark.parametrize("I", (10, 100, 1000))
def test_NPSamplers(
    a_dist: NPDistribution, b_dist: NPDistribution, V_dist: NPDistribution, I: int  # noqa: E741
) -> None:
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b is not None and b.shape == (I,)
    assert V.shape == (I, 2)
