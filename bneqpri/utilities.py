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

import numpy as np

# from typing import Optional

from .types import NPArrayType
from .samplers import ABVSampler


class Utility:
    r"""A mixture-of-logits-like model for pricing can be considered to be
    any model with a structure of the form

        u_{i,j}(p_j) = a_i f(b_i;p_j) + v_{i,j}

    for some function f of a budget b_i and price p_j. We can consider a_i
    to be the price sensitivity, and v_{i,j} the non-price utility. Formally,
    a_i, b_i could depend on both individual i and product j, resulting in
    very similar calculations. But because that is perhaps over-parameterized,
    and parsimony can help generalizability, we do not model for this here.

    This class just serves as an "interface" definition. Subclasses fill out
    actual details of an implementation of f and it's derivatives (which form the
    utility-price derivatives needed to compute equilibrium prices).

    Any utility will need to hold data for the three terms above,

        a: an I-element vector ("drawn" from a distribution)
        b: an I-element vector ("drawn" from a distribution)
        V: an I x J matrix ("drawn" from a distribution)

    Note that V can be seen as a function of "latent" (to pricing) product
    characteristics X (some J x K matrix) in this model, as in a linear model

        v_{i,j} = sum_k w_{i,k} x_{k,j}

    (where the X's may also effect costs). This does not effect the pricing model
    beyond the influence of V (and/or c), so we leave it up to the user to determine
    these effects.

    The fundamental operations required of a utility are (i) sampling a, b, and V
    and (ii) computing associated values for U, DpU, and possibly DppU (utilities
    and their first and second derivatives). The first operation is supplied by
    an "ABVSampler" (see types.py and samplers.py): any Callable that takes in a
    sample size and returns suitable a, b, and V.
    """

    a: NPArrayType
    b: NPArrayType
    V: NPArrayType

    maxbi: int
    maxb: float

    U: NPArrayType
    DpU: NPArrayType
    DppU: NPArrayType

    def __init__(
        self,
        products: int,  # number of products
        sampler: ABVSampler,
    ) -> None:

        self.J: int = products  # number of products

        self.I: int = 0  # noqa: E741
        # self.U: Optional[NPArrayType] = None
        # self.DpU: Optional[NPArrayType] = None
        # self.DppU: Optional[NPArrayType] = None

        self.sampler = sampler

        # # basic utility parameters (for `a f(b;p) + v` form)
        # self.a: Optional[NPArrayType] = None  # I vector when sampled
        # self.b: Optional[NPArrayType] = None  # I vector when sampled
        # self.V: Optional[NPArrayType] = None  # I x J matrix when sampled

        # max budget and its index, if there are budgets (for "correction")
        # self.maxbi: Optional[int] = None
        # self.maxb: Optional[float] = None

    def sample(self, individuals: int) -> None:  # noqa: E741
        r"""Sample utility parameters a, b, and V using supplied sampler."""

        # set sample size
        self.I = individuals  # noqa: E741

        # 3IxJ storage for utilities and first and second price derivatives
        #
        #     du_{i,j}/dp (p_j) = a_i df/dp(b_i;p_j)
        #     d^2u_{i,j}/dp^2 (p_j) = a_i d^2f/dp^2(b_i;p_j)
        #
        self.U = np.zeros((self.I, self.J))  # utilities (for each individual/product)
        self.DpU = np.zeros((self.I, self.J))  # price derivatives of utilities (for each)
        self.DppU = np.zeros((self.I, self.J))  # second price derivatives of utilities (fe)

        # sample from Dist(a, b, <v>) setting self.a, self.b, self.V
        self.a, self.b, self.V = self.sampler(self.I)

        # assert: a ~ I x 1, b ~ I x 1 or None, V ~ I x J
        # The following _should_ raise errors if the arrays
        # cannot be reshaped correctly

        self.a = self.a.reshape(self.I, 1)

        if self.b is not None:
            self.b = self.b.reshape(self.I, 1)
            self.maxbi = int(np.argmax(self.b))  # used in corrected iterations
            self.maxb = self.b[self.maxbi]

        self.V = self.V.reshape(self.I, self.J)

    def __call__(self, p: NPArrayType) -> None:
        raise NotImplementedError("Base utility class does not implement itself")

        # TODO: implement shift logic for exp float fix
        # self.umax[:] = np.maximum(0, np.max(self.U, axis=1)) # I vector
        # self.U = self.U - self.umax[:I,] * ones[:J,]'


class LinearUtility(Utility):
    def __call__(self, p: NPArrayType) -> None:
        self.U = self.a * p.reshape(1, self.J) + self.V
        self.DpU = self.a * np.ones((1, self.J))
        # self.DppU = np.zeros((I,J)) default, also not needed


class LogOfRemainingUtility(Utility):
    def sample(self, individuals: int) -> None:  # noqa: E741
        super().sample(individuals)
        self.A = self.a * np.ones((1, self.J))  # I x J column replication

    def __call__(self, p: NPArrayType) -> None:
        """For a[i] > 1, if p[j] < b[i] set

               U[i,j] =   a[i] log(b[i]-p) + V[i,j]
             DpU[i,j] = - a[i] / (b[i]-p)    (-> -Inf as p -> b[i])
            DppU[i,j] = - a[i] / (b[i]-p)^2  (-> -Inf as p -> b[i])

        for "budget" b[i] or

               U[i,j] ~ - 1.0e20
             DpU[i,j] = - 0.0
            DppU[i,j] = - 1/a[i]
                      = lim_{ p -> b[i] } [ DppU[i,j](p) / DpU[i,j](p)^2 ]

        if p[j] >= b[i] (we _store_ a ratio in DppU, not the actual value).

        This weird storage format is just because we need that
        limiting ratio of first/second derivatives for the right
        extension of the fixed point map. See the paper.

        The setting of DpU[i,j] to zero is basically an expression
        of the condition DpU[i,j] PL[i,j] -> 0 as p[j] -> b[i].
        This is discussed in Assumption 1 of the paper, but
        basically ensures sufficient continuity for use of deriv-
        atives for analyzing equilibrium. By setting DpU[i,j] to
        zero, we avoid numerical issues from other values.

        Technically, we should probably build some confidence that
        values like DpU[i,j] PL[i,j] are _numerically_ continuous
        as well.

        """
        T = self.b - p.reshape(1, self.J)  # type: ignore
        T0 = np.where(T <= 0)  # boolean I x J mask

        self.U = self.A * np.log(T) + self.V  # U[i,j] == nan if T[i,j] <= 0
        self.DpU = -self.A / T
        self.DppU = self.DpU / T

        self.U[T0] = -1.0e20  # effectively negative infinity
        self.DpU[T0] = 0.0  # effective limit as p[j] -> b[i] in P, Lam, Gam
        self.DppU[T0] = -1.0 / self.A[T0]  # correction, actually DppU[i,j]/DpU[i,j]^2

        # # TODO: very inefficient double loop; how to eliminate?
        # for i in range(self.I):
        #     for j in range(self.J):
        #         if T[i, j] > 0.0:  # tolerance, not just > 0.0?
        #             self.U[i, j] = self.a[i] * np.log(T[i, j]) + self.V[i, j]
        #             self.DpU[i, j] = -self.a[i] / T[i, j]
        #             self.DppU[i, j] = self.DpU[i, j] / T[i, j]
        #         else:
        #             self.U[i, j] = -1.0e20
        #             self.DpU[i, j] = 0.0
        #             self.DppU[i, j] = -1.0 / self.a[i]


class ReciprocalOfRemainingUtility(Utility):
    def sample(self, individuals: int) -> None:  # noqa: E741
        super().sample(individuals)
        self.A = self.a * np.ones((1, self.J))  # I x J column replication

    def __call__(self, p: NPArrayType) -> None:
        """For any a[i] < 0

               U[i,j] = a[i] / (b[i]-p)      (-> -Inf as p -> b[i])
             DpU[i,j] = a[i] / (b[i]-p)^2    (-> -Inf as p -> b[i]) ==> a[i] < 0 required
            DppU[i,j] = 2 a[i] / (b[i]-p)^3  (-> -Inf as p -> b[i])

        when p < b[i] but

               U[i,j] = -1.0e20
             DpU[i,j] = 0.0
            DppU[i,j] ~ lim_{ p -> b[i] } [ DppU[i,j](p) / DpU[i,j](p)^2 ]
                      = lim_{ p -> b[i] } [ 2 a[i] / (b[i]-p)^3 ] / [ a[i] / (b[i]-p)^2 ]^2
                      = 2 lim_{ p -> b[i] } [ a[i] (b[i]-p)^4 ] / [ a[i]^2 (b[i]-p)^3 ]
                      = 2 lim_{ p -> b[i] } (b[i]-p) / a[i] = 0

        when p >= b[i] (we _store_ a ratio in DppU, not the actual value).
        """
        T = self.b - p.reshape(1, self.J)  # type: ignore
        T0 = np.where(T <= 0)  # boolean I x J mask

        AoT = self.A / T
        self.U = AoT + self.V
        self.DpU = AoT / T
        self.DppU = 2.0 * self.DpU / T

        self.U[T0] = -1.0e20  # effectively negative infinity
        self.DpU[T0] = 0.0  # effective limit as p[j] -> b[i] in P, Lam, Gam
        self.DppU[T0] = 0.0  # correction, actually DppU[i,j]/DpU[i,j]^2

        # # TODO: very inefficient double loop; how to eliminate?
        # for i in range(self.I):
        #     for j in range(self.J):
        #         if T[i, j] > 0.0:  # tolerance, not just > 0.0?
        #             self.U[i, j] = self.a[i] / T[i, j] + self.V[i, j]
        #             self.DpU[i, j] = self.a[i] / (T[i, j] * T[i, j])
        #             self.DppU[i, j] = 2.0 * self.DpU[i, j] / T[i, j]
        #         else:
        #             self.U[i, j] = -1.0e20
        #             self.DpU[i, j] = 0.0
        #             self.DppU[i, j] = 0.0
