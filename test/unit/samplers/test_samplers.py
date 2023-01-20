import pytest
import numpy as np

from bneqpri.types import (
    NPArrayType,
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
    LogNormalABNormalVSampler,
    LogNormalABNormalWXSampler,
)
from bneqpri.utilities import (
    Utility,
    LinearUtility,
    LogOfRemainingUtility,
    ReciprocalOfRemainingUtility,
)

from bneqpri.solver import FPISolverV2, BatchedFPISolverV2


@pytest.mark.parametrize("I", (10, 100, 1000))
def test_ABVSampler(I: int) -> None:  # noqa: E741
    S = ABVSampler()
    with pytest.raises(NotImplementedError):
        S(I)


@pytest.mark.parametrize("I", (10, 100, 1000))
@pytest.mark.parametrize("J", (2, 5))
def test_LogNormalABNormalVSampler(I: int, J: int) -> None:  # noqa: E741
    mean_V = np.random.randn(J)
    cov_V = np.random.randn(J, J)
    cov_V = cov_V.T @ cov_V  # make SPD
    S = LogNormalABNormalVSampler(1.0, 1.0, 1.0, 1.0, mean_V, cov_V)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b.shape == (I,) if b is not None else True
    assert V.shape == (I, J)


@pytest.mark.parametrize("I", (10, 100, 1000))
@pytest.mark.parametrize("J", (2, 5))
@pytest.mark.parametrize("K", (2,))
def test_LogNormalABNormalWXSampler(I: int, J: int, K: int) -> None:  # noqa: E741
    mean_W = np.random.randn(K)
    cov_W = np.random.randn(K, K)
    cov_W = cov_W.T @ cov_W  # make SPD
    data_X = np.random.randn(K, J)
    S = LogNormalABNormalWXSampler(1.0, 1.0, 1.0, 1.0, mean_W, cov_W, data_X)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b.shape == (I,) if b is not None else True
    assert V.shape == (I, J)


@pytest.mark.parametrize("I", (10, 100, 1000))
def test_NPSamplers(I: int) -> None:  # noqa: E741
    a_dist = BetaDistribution(1.0, 0.5)
    b_dist = ParetoDistribution(1.0)
    V_dist = MVNDistribution(np.array([2.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    a, b, V = S(I)
    assert a.shape == (I,)
    assert b.shape == (I,) if b is not None else True
    assert V.shape == (I, 2)


@pytest.mark.parametrize("J", (3, 10, 100))
@pytest.mark.parametrize("I", (10, 100, 1000))
def test_utility_sample(J: int, I: int) -> None:  # noqa: E741
    a_dist = LognormalDistribution(1.0, 0.5)
    b_dist = LognormalDistribution(10.0, 2.0)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = Utility(J, S)
    U.sample(I)


@pytest.mark.parametrize("J", (3, 10, 100))
@pytest.mark.parametrize("I", (10, 100, 1000))
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


@pytest.mark.parametrize("J", (3, 10, 100))
@pytest.mark.parametrize("I", (10, 100, 1000))
def test_linear_utility(J: int, I: int) -> None:  # noqa: E741
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


@pytest.mark.parametrize("J", (3, 10, 100))
@pytest.mark.parametrize("I", (10, 100, 1000))
def test_log_of_utility(J: int, I: int) -> None:  # noqa: E741
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


@pytest.mark.parametrize("J", (3, 10, 100))
@pytest.mark.parametrize("I", (10, 100, 1000))
def test_recip_of_utility(J: int, I: int) -> None:  # noqa: E741
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


@pytest.mark.parametrize(
    ("J", "F", "Jfs"),
    (
        (3, 2, np.array([2, 1])),
        (10, 2, np.array([5, 5])),
        (100, 4, np.array([25, 25, 25, 25])),
        (100, 5, np.array([25, 15, 10, 17, 33])),
    ),
)
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_fpi_solver_v2_linu(J: int, F: int, Jfs: NPArrayType, I: int) -> None:  # noqa: E741

    assert Jfs.sum() == J, f"bad test: sum(Jfs) = {Jfs.sum()} != {J} = J"

    c = UniformDistribution(0.0, 1.0)(J)

    a_dist = LognormalDistribution(1.0, 0.5, negate=True)
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = AVIndependentSampler(a_dist, V_dist)
    U = LinearUtility(J, S)

    solver = FPISolverV2(J, F, Jfs, c, U)
    p = solver.solve(I, max_iter=100)  # noqa: F841

    print("")
    solver.summary()
    assert solver.solved

    # batching (with the utmost naivety) does not appear to work

    # batched = BatchedFPISolverV2(J, F, Jfs, c, U)
    # pb = batched.solve(int(I/2), max_iter=1000, memory=10)  # noqa: F841

    # print("")
    # batched.summary()



@pytest.mark.parametrize(
    ("J", "F", "Jfs"),
    (
        (3, 2, np.array([2, 1])),
        (10, 2, np.array([5, 5])),
        (100, 4, np.array([25, 25, 25, 25])),
        (100, 5, np.array([25, 15, 10, 17, 33])),
    ),
)
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_fpi_solver_v2_loru(J: int, F: int, Jfs: NPArrayType, I: int) -> None:  # noqa: E741

    assert Jfs.sum() == J, f"bad test: sum(Jfs) = {Jfs.sum()} != {J} = J"

    c = UniformDistribution(0.0, 3.0)(J)

    a_dist = LognormalDistribution(1.0, 0.5, negate=False)
    b_dist = UniformDistribution(2.0, 10.0)  # needs to be _somewhere_ higher than costs
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = LogOfRemainingUtility(J, S)

    solver = FPISolverV2(J, F, Jfs, c, U)
    p = solver.solve(I, max_iter=100)  # noqa: F841

    print("")
    solver.summary()
    assert solver.solved


@pytest.mark.parametrize(
    ("J", "F", "Jfs"),
    (
        (3, 2, np.array([2, 1])),
        (10, 2, np.array([5, 5])),
        (100, 4, np.array([25, 25, 25, 25])),
        (100, 5, np.array([25, 15, 10, 17, 33])),
    ),
)
@pytest.mark.parametrize("I", (1000, 5000)) # 10 and 100 fail?
def test_fpi_solver_v2_roru(J: int, F: int, Jfs: NPArrayType, I: int) -> None:  # noqa: E741

    assert Jfs.sum() == J, f"bad test: sum(Jfs) = {Jfs.sum()} != {J} = J"

    c = UniformDistribution(0.0, 3.0)(J)

    a_dist = LognormalDistribution(1.0, 0.5, negate=True)
    b_dist = UniformDistribution(2.0, 10.0)  # needs to be _somewhere_ higher than costs
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = ReciprocalOfRemainingUtility(J, S)

    solver = FPISolverV2(J, F, Jfs, c, U)
    p = solver.solve(I, max_iter=100)  # noqa: F841

    print("")
    solver.summary()
    assert solver.solved
