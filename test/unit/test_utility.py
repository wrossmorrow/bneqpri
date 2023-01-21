import pytest
import numpy as np

from bneqpri.types import (
    MVNDistribution,
    LognormalDistribution,
    UniformDistribution,
)
from bneqpri.samplers import (
    ABVIndependentSampler,
    AVIndependentSampler,
)
from bneqpri.utilities import (
    Utility,
    LinearUtility,
    LogOfRemainingUtility,
    ReciprocalOfRemainingUtility,
)


@pytest.mark.parametrize("J", (3, 10, 100, 1000))
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_utility_sample(J: int, I: int) -> None:  # noqa: E741
    a_dist = LognormalDistribution(1.0, 0.5)
    b_dist = LognormalDistribution(10.0, 2.0)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = Utility(J, S)
    U.sample(I)


@pytest.mark.parametrize("J", (3, 10, 100, 1000))
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_utility_no_compute(J: int, I: int) -> None:  # noqa: E741
    a_dist = LognormalDistribution(1.0, 0.5)
    b_dist = LognormalDistribution(10.0, 2.0)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = Utility(J, S)
    U.sample(I)
    p = UniformDistribution(1.0, 2.0)(J)
    with pytest.raises(NotImplementedError):
        U(p)


@pytest.mark.parametrize("J", (3, 10, 100, 1000))
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_utility_linu(J: int, I: int) -> None:  # noqa: E741
    a_dist = LognormalDistribution(1.0, 0.5)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = AVIndependentSampler(a_dist, V_dist)
    U = LinearUtility(J, S)
    U.sample(I)
    p = UniformDistribution(1.0, 2.0)(J)
    U(p)
    assert U.U.shape == (I, J)
    assert U.DpU.shape == (I, J)
    assert U.DppU.shape == (I, J)


@pytest.mark.parametrize("J", (3, 10, 100, 1000))
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_utility_loru(J: int, I: int) -> None:  # noqa: E741
    a_dist = LognormalDistribution(1.0, 0.5)
    b_dist = LognormalDistribution(10.0, 2.0)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = LogOfRemainingUtility(J, S)
    U.sample(I)
    p = UniformDistribution(1.0, 2.0)(J)
    U(p)
    assert U.U.shape == (I, J)
    assert U.DpU.shape == (I, J)
    assert U.DppU.shape == (I, J)

    for i in range(I):
        for j in range(J):
            if p[j] > U.b[i]:
                assert U.U[i, j] <= -1.0e20
                assert U.DpU[i, j] == 0.0
                assert U.DppU[i, j] == -1.0 / U.a[i]
            else:
                assert U.U[i, j] is not np.nan
                assert U.DpU[i, j] is not np.nan
                assert U.DppU[i, j] is not np.nan


@pytest.mark.parametrize("J", (3, 10, 100, 1000))
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_utility_roru(J: int, I: int) -> None:  # noqa: E741
    a_dist = LognormalDistribution(1.0, 0.5)
    b_dist = LognormalDistribution(10.0, 2.0)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = ReciprocalOfRemainingUtility(J, S)
    U.sample(I)
    p = UniformDistribution(1.0, 2.0)(J)
    U(p)
    assert U.U.shape == (I, J)
    assert U.DpU.shape == (I, J)
    assert U.DppU.shape == (I, J)

    for i in range(I):
        for j in range(J):
            if p[j] > U.b[i]:
                assert U.U[i, j] <= -1.0e20
                assert U.DpU[i, j] == 0.0
                assert U.DppU[i, j] == 0.0
            else:
                assert U.U[i, j] is not np.nan
                assert U.DpU[i, j] is not np.nan
                assert U.DppU[i, j] is not np.nan
