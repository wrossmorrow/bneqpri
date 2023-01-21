from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from numpy.random import default_rng


# TODO: figure out numpy nd-array typing (? ndarray[Any, dtype[floating[_64Bit]]])
NPArrayType = Any

# Samples output: a[:I], b[:I], V[:I,:J]
ABVSamples = Tuple[NPArrayType, Optional[NPArrayType], NPArrayType]

# The allowable type of a sampler callable - accept a size (number of
# individuals, "I") and return samples from
ABVSamplerCall = Callable[[int], ABVSamples]

ABVMarginal = Callable[[int], NPArrayType]


@dataclass
class Market:
    products: int  # number of products
    firms: int  # number of firms
    products_per_firm: Union[
        List[int], NPArrayType
    ]  # number of products per firm, expects F elements


class NPDistribution:
    def __init__(
        self, name: str, params: Tuple[Any, ...], negate: bool = False, shift: float = 0.0
    ) -> None:
        self._name = name
        self._params = params
        self._sign = -1 if negate else 1
        self._shift = shift
        self._rng = default_rng()
        assert hasattr(self._rng, self._name)
        self._call = getattr(self._rng, self._name)

    def __call__(self, I: int) -> ABVSamples:  # noqa: #741
        return self._sign * self._call(*self._params, size=I) + self._shift


class NormalDistribution(NPDistribution):
    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__("normal", (mu, sigma))


class MVNDistribution(NPDistribution):
    def __init__(self, mu: NPArrayType, cov: NPArrayType) -> None:
        super().__init__("multivariate_normal", (mu, cov))


class LognormalDistribution(NPDistribution):
    def __init__(self, mu: float, sigma: float, negate: bool = False, shift: float = 0.0) -> None:
        super().__init__("lognormal", (mu, sigma), negate=negate, shift=shift)


class BetaDistribution(NPDistribution):
    def __init__(self, a: float, b: float, negate: bool = False) -> None:
        super().__init__("beta", (a, b), negate=negate)


class Chi2Distribution(NPDistribution):
    def __init__(self, dfs: Tuple[float], negate: bool = False) -> None:
        super().__init__("chisquare", dfs, negate=negate)


class GumbelDistribution(NPDistribution):
    def __init__(self, loc: float, scale: float, negate: bool = False) -> None:
        super().__init__("gumbel", (loc, scale), negate=negate)


class LogisticDistribution(NPDistribution):
    def __init__(self, loc: float, scale: float, negate: bool = False) -> None:
        super().__init__("logistic", (loc, scale), negate=negate)


class ParetoDistribution(NPDistribution):
    def __init__(self, a: float, negate: bool = False) -> None:
        super().__init__("pareto", (a,), negate=negate)


class PowerDistribution(NPDistribution):
    def __init__(self, a: float, negate: bool = False) -> None:
        super().__init__("power", (a,), negate=negate)


class TriangularDistribution(NPDistribution):
    def __init__(self, left: float, mode: float, right: float) -> None:
        super().__init__("triangular", (left, mode, right))


class UniformDistribution(NPDistribution):
    def __init__(self, low: float, high: float) -> None:
        super().__init__("uniform", (low, high))
