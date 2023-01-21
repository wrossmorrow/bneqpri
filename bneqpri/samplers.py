"""

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Copyright W. Ross Morrow (morrowwr@gmail.com), 2020+

"""
from typing import Optional, Union

import numpy as np

from .types import (
    NPArrayType,
    ABVMarginal,
    ABVSamples,
    NPDistribution,
    LognormalDistribution,
    MVNDistribution,
)


class ABVSampler:
    r"""The allowable type of a sampler callable - accept a size (number of
    individuals, "I"), sample values for

            (a, b, V) ~ some distribution
            a is an I vector
            b is an I vector
            V is an I x J matrix

    J must be "implicitly" defined by the sample consistent with a market
    structure, as can be
    """

    def __init__(self) -> None:
        pass

    def __call__(self, I: int) -> ABVSamples:  # noqa: E741
        raise NotImplementedError(f"{self.__class__.__name__} does not implement sampling")


class ABVIndependentSampler(ABVSampler):
    r"""Any independent sampler"""

    def __init__(
        self,
        a_sampler: Union[ABVMarginal, NPDistribution],
        b_sampler: Optional[Union[ABVMarginal, NPDistribution]],
        V_sampler: Union[ABVMarginal, NPDistribution],
    ) -> None:
        self._a_sampler = a_sampler
        self._b_sampler = b_sampler
        self._V_sampler = V_sampler

    def __call__(self, I: int) -> ABVSamples:  # noqa: #741
        return (
            self._a_sampler(I),
            self._b_sampler(I) if self._b_sampler else None,
            self._V_sampler(I),
        )


class AVIndependentSampler(ABVIndependentSampler):
    """Any independent sampler with no budgets"""

    def __init__(
        self,
        a_sampler: Union[ABVMarginal, NPDistribution],
        V_sampler: Union[ABVMarginal, NPDistribution],
    ) -> None:
        super().__init__(a_sampler, None, V_sampler)


class LognormalABNormalVSampler(ABVIndependentSampler):
    """Log normal a and b, but normal V. a coefficients may be negated.


    This is a useful base distribution class because a and b coefficients
    are implicitly "signed" to have a well-formed equilibrium pricing problem.
    Particularly, we should have

        b_i > 0, sign{a_i} == - sign{f(b_i,•)} (so du/dp < 0)

    (or if b_i is not > 0, f effectively inverts it's sign).
    """

    def __init__(
        self,
        logmean_a: float,
        logsigma_a: float,
        logmean_b: float,
        logsigma_b: float,
        mean_V: NPArrayType,  # expects J vector
        cov_V: NPArrayType,  # expects J x J SPD matrix
        negate_a: bool = False,
    ) -> None:
        a_dist = LognormalDistribution(logmean_a, logsigma_a, negate=negate_a)
        b_dist = LognormalDistribution(logmean_b, logsigma_b)
        V_dist = MVNDistribution(mean_V, cov_V)
        super().__init__(a_dist, b_dist, V_dist)


class LognormalABNormalWXSampler(ABVIndependentSampler):
    """Log normal a and b, but normal V derived from some data matrix X
    of "product characteristics" and a multivariate normal weight matrix.
    """

    def __init__(
        self,
        logmean_a: float,
        logsigma_a: float,
        logmean_b: float,
        logsigma_b: float,
        mean_W: NPArrayType,  # expects K vector
        cov_W: NPArrayType,  # expects K x K SPD matrix
        data_X: NPArrayType,  # expects K x J matrix
        negate_a: bool = False,
    ) -> None:

        K, J = data_X.shape  # expansion will fail if #dims(data_X) != 2
        assert mean_W.shape[0] == K
        assert cov_W.shape[0] == K and cov_W.shape[1] == K

        a_dist = LognormalDistribution(logmean_a, logsigma_b, negate=negate_a)
        b_dist = LognormalDistribution(logmean_b, logsigma_b)

        # V is effectively J independent normal distributions with means
        # and variances determined by the affine transformation
        #
        #   v_{i,j} = W_i' X[:,j] = X[:,j]' W_i ~ N(X[:,j]'mu_W, X[:,j]' cov_W X[:,j])
        #
        mean_V = data_X.T.dot(mean_W)
        cov_V = np.diag(np.array([data_X[:, j].T.dot(cov_W.dot(data_X[:, j])) for j in range(J)]))
        V_dist = MVNDistribution(mean_V, cov_V)

        super().__init__(a_dist, b_dist, V_dist)
