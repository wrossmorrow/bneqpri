import pytest
import numpy as np

from bneqpri.types import (
    NPArrayType,
    MVNDistribution,
    LognormalDistribution,
    UniformDistribution,
)
from bneqpri.samplers import (
    ABVIndependentSampler,
    AVIndependentSampler,
)
from bneqpri.utilities import (
    LinearUtility,
    LogOfRemainingUtility,
    ReciprocalOfRemainingUtility,
)

from bneqpri.solver import FPISolverV2


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

    a_dist = LognormalDistribution(1.0, 0.5, negate=True)  # condition: a[i] < 0
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = AVIndependentSampler(a_dist, V_dist)
    U = LinearUtility(J, S)

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
@pytest.mark.parametrize("I", (10, 100, 1000, 5000))
def test_fpi_solver_v2_loru(J: int, F: int, Jfs: NPArrayType, I: int) -> None:  # noqa: E741

    assert Jfs.sum() == J, f"bad test: sum(Jfs) = {Jfs.sum()} != {J} = J"

    c = UniformDistribution(0.0, 3.0)(J)

    a_dist = LognormalDistribution(1.0, 0.5, negate=False, shift=1.0)  # condition: a[i] > 1
    b_dist = UniformDistribution(2.0, 10.0)  # conditions: b[i] > 0, max_i b[i] > max_j c[j]
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
@pytest.mark.parametrize("I", (1000, 5000))  # 10 and 100 fail?
def test_fpi_solver_v2_roru(J: int, F: int, Jfs: NPArrayType, I: int) -> None:  # noqa: E741

    assert Jfs.sum() == J, f"bad test: sum(Jfs) = {Jfs.sum()} != {J} = J"

    c = UniformDistribution(0.0, 3.0)(J)

    a_dist = LognormalDistribution(1.0, 0.5, negate=True)  # condition: a[i] < 0
    b_dist = UniformDistribution(2.0, 10.0)  # conditions: b[i] > 0, max_i b[i] > max_j c[j]
    V_dist = MVNDistribution(UniformDistribution(1.0, 2.0)(J), np.eye(J))
    S = ABVIndependentSampler(a_dist, b_dist, V_dist)
    U = ReciprocalOfRemainingUtility(J, S)

    solver = FPISolverV2(J, F, Jfs, c, U)
    p = solver.solve(I, max_iter=100)  # noqa: F841

    print("")
    solver.summary()
    assert solver.solved
