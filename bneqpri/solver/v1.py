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

from time import time
from typing import List, Optional, Tuple, Union

from ..types import NPArrayType


class FPISolver:
    def __init__(
        self,
        I: int,  # noqa: E741  # TODO: remove when specifying distributions
        J: int,
        F: int,
        Jf: NPArrayType,  # expects F elements
        Bi: Optional[NPArrayType],  # expects I elements
        a: NPArrayType,  # expects I elements
        V: NPArrayType,  # expects I x J matrix
        c: NPArrayType,  # expects J elements
    ) -> None:
        """
        I:  number of individuals (integer)
        J:  number of products (integer)
        F:  number of firms (integer)
        Jf: number of products per firm (F vector)
        Bi: "budgets" for each individual (I vector)
        a:  price sensitivity for each individual (I vector)
        V:  nonprice utility for each individual/product (I x J matrix)
        c:  costs for each product (J vector)
        """

        # problem data
        self.I = I  # noqa: E741
        self.J = J  # number of products
        self.F = F  # number of firms
        self.Jf = Jf  # list of the number of products per firm
        self.b = Bi  # individual incomes
        self.a = a  # price sensitivity
        self.V = V  # "fixed" portion of utility (non-price part)
        self.c = c  # (unit) costs for each product

        # internal data

        # max income and max income index
        if Bi is not None:
            self.maxbi = np.argmax(Bi)
            self.maxb = Bi[self.maxbi]

        # indices segmenting firm blocks
        self.Fis: List[List[int]] = [[] for f in range(F)]
        self.Fis[0] = list(range(0, self.Jf[0]))
        for f in range(1, F):
            self.Fis[f] = list(
                range(self.Fis[f - 1][-1] + 1, self.Fis[f - 1][-1] + 1 + self.Jf[f])
            )

        # other storage
        self.m = np.zeros(J)  # markups (prices minus costs)
        self.pr = np.zeros(self.F)  # profits (convenience only)
        self.P = np.zeros(J)  # mixed probabilities
        self.L = np.zeros(J)  # "Lambda" operator, diagonal matrix
        self.G = np.zeros((J, J))  # "Gamma" operator, dense matrix
        self.z = np.zeros(J)  # "zeta" map
        self.phi = np.zeros(J)  # "phi" map, what we want to zero

        self.stats: List[List[Union[int, float]]] = []  # solve iteration "statistics"

        # TODO: migrate to either a "utility" class and/or solve-specific
        # sampling scheme by specifying the distribution of alpha, beta, value
        # (price sensitivity, budget, "cummulative part worth")

        self.U = np.zeros((I, J))  # utilities (for each individual/product)
        self.DpU = np.zeros((I, J))  # price derivatives of utilities (for each)
        self.DppU = np.zeros((I, J))  # second price derivatives of utilities (fe)
        self.PL = np.zeros((I, J))  # logit probabilities (idiosyncratic)

    def solve(
        self,
        p0: Optional[NPArrayType] = None,
        f_tol: float = 1.0e-6,
        max_iter: int = 1000,
        corrected: bool = True,
        batched: bool = False,
        verbose: bool = False,
        check: bool = False,
    ) -> NPArrayType:

        # TODO: Use distributions to characterize (alpha, beta, value), and
        # draw here for a specified value of I or otherwise an increasing
        # sequence of I values
        #
        #     I = <some integer, possibly passed in>
        #
        # draw values for
        #
        #     self.b = Bi  # individual budgets
        #     self.a = a   # price sensitivity
        #     self.V = V   # "fixed" portion of utility (non-price part)
        #
        # from values past in as distributions. This is also relevant if correcting:
        #
        #     # max income and max income index
        #     self.maxbi = np.argmax(self.b)
        #     self.maxb = self.b[self.maxbi]
        #
        # Initialize
        #
        #     self.U = np.zeros((I, J))  # utilities (for each individual/product)
        #     self.DpU = np.zeros((I, J))  # price derivatives of utilities (for each)
        #     self.DppU = np.zeros((I, J))  # second price derivatives of utilities (fe)
        #     self.PL = np.zeros((I, J))  # logit probabilities (idiosyncratic)
        #

        # random prices in [ c/2 , 3/2c ] if not specified
        p = p0 if p0 is not None else (self.c / 2.0 + 2.0 * np.random.random(self.J))

        step = self.zeta_c if corrected else self.zeta_u

        self.nrms = np.zeros(max_iter)
        self.solved = False
        self.stats = []
        start = time()
        for self.iter in range(max_iter):

            if check:
                self.probcheck(p)
                self.gradcheck(p)

            # compute "step", ie the (corrected?) zeta map
            step(p, verbose=verbose)

            # test convergence (using step, not combined gradient)
            self.nrms[self.iter] = np.max(np.abs(self.phi))

            self.stats.append(
                [
                    time() - start,
                    self.iter,
                    p.min(),
                    p.max(),
                    self.pr.min(),
                    self.pr.max(),
                    self.P.sum(),
                    self.nrms[self.iter],
                ]
            )

            if verbose:
                self.profits(p)
                self._progress(p)

            if self.nrms[self.iter] <= f_tol:
                self.solved = True
                break

            # fixed-point step, equivalently p -> p - phi = p - ( p - c - z )
            p = self.c + self.z

        self.time = time() - start

        self.nrms = self.nrms[: self.iter + 1]

        return p

    def _progress(self, p: NPArrayType) -> None:
        print(", ".join([str(s) for s in self.stats[-1]]))  # type: ignore
        #         print(
        #             f"""
        # Iteration {self.iter}:
        #   min/max price..... {p.min()}, {p.max()}
        #   min/max profits... {self.pr.min()}, {self.pr.max()}
        #   marketshare....... {self.P.sum()}
        #   phi norm.......... {self.nrms[self.iter]}
        # """
        #         )

    def zeta_u(self, p: NPArrayType, verbose: bool = False) -> None:
        """Uncorrected "zeta map"

        z <- inv(diag(LAMp)) * ( \tilde{GAMp}' * m - P )

        """
        self._iterprep(p)  # utiliites, probabilities, lambda, gamma, markups
        self._zeta_b()  # the "base" part of either zeta map
        self.z = self.z / self.L  # z <- inv(diag(LAMp)) * z
        self.phi = self.m - self.z  # phi map (self.m updated with _iterprep)

    def zeta_c(self, p: NPArrayType, verbose: bool = False):
        """Corrected "zeta map"

        z <- inv(diag(LAMp)) * ( \tilde{GAMp}' * m - P )
        for all prices < maxinc, corrected otherwise

        """
        self._iterprep(p)  # utiliites, probabilities, lambda, gamma, markups
        self._zeta_b()  # the "base" part of either zeta map

        # nominally z <- inv(diag(LAMp)) * z, but with corrections
        # for products whose prices are above the population limit
        # on incomes. The correction is
        #
        #     z[j] = omega[maxinci,j] * ( p[j] - maxinc ) + PL[maxinci,{f}]' * m[{f}]
        #
        # for j : p[j] > maxb
        corr_idx: List[Tuple[int, int]] = []
        for f in range(self.F):
            fi = self.Fis[f]
            prFmi = self.PL[self.maxbi, fi].T.dot(self.m[fi])
            for j in fi:
                if p[j] > self.maxb:  # correction term - price j is too high
                    corr_idx.append((f, j))
                    # print(
                    #     f"WARNING ({self.iter}) -- {p[j]} > {self.maxb} correcting zeta for product {j}"
                    # )
                    self.z[j] = self.DppU[self.maxbi, j] * (p[j] - self.maxb) + prFmi
                elif self.L[j] < 0.0:
                    # some tolerance would be better than "0", like
                    # LAM[j] < -1.0e-10 (just as an example)
                    self.z[j] /= self.L[j]
                else:
                    # here self.L ~ 0.0, i.e. PL ~ 0.0
                    print(
                        f"WARNING ({self.iter}) -- p[{j}] = {p[j]} < {self.maxb} = maxinc but LAMp[{j}] = {self.L[j]}."
                    )
                    # use a modification of extended map instead of what is calculated above
                    # z[j] = PL[maxinci,{f}]' * m[{f}]
                    self.z[j] = prFmi
                    # we exclude the "DppU[I*j+maxinci] * ( p[j] - maxinc )" term expecting
                    # p[j] to be at least close to maxinc

        if verbose:
            print(f"Corrected: {corr_idx}")

        # compute phi = p - c - z also (self.m updated with _iterprep)
        self.phi = self.m - self.z

    def _iterprep(self, p: NPArrayType) -> None:
        self.utilities(p)
        self.probabilities()
        self.lamgam()
        self.m = p - self.c

    def _zeta_b(
        self,
    ) -> None:
        """z <- \tilde{GAMp}' * m - P"""
        for f in range(self.F):
            fi = self.Fis[f]
            # self.z[fi] = self.G[fi,fi].T.dot(self.m[fi]) - self.P[fi]
            self.z[fi] = -self.P[fi]
            for j in fi:
                for k in fi:
                    self.z[j] += self.G[k, j] * self.m[k]

    def utilities(self, p: NPArrayType) -> None:
        raise NotImplementedError(
            "The method `utilities` is not implemented in the base class. "
            "Please use or define a subclass that implements this method."
        )

    def probabilities(self) -> None:
        self.PL, self.P = 0.0 * self.PL, 0.0 * self.P  # zero out, not reallocate
        uimax = np.maximum(0, np.max(self.U, axis=1))  # exp float fix
        for i in range(self.I):  # TODO: large loop, figure out broadcast mechanism
            self.PL[i, :] = np.exp(self.U[i, :] - uimax[i])
            self.PL[i, :] /= np.exp(-uimax[i]) + self.PL[i, :].sum()
        self.P = np.sum(self.PL, axis=0) / self.I

    def lamgam(self) -> None:
        """Compute the "Lambda" and "Gamma" matrices from the papers. Note
        that "Lambda" (`L`) is a diagonal matrix, but "Gamma" (`G`) is full.

        Also note some theory:

            DpU * PL -> 0 as p -> b

        so long as

            - DppU/(DpU)^2 is bounded as p -> b

        This follows from L'Hopital's rule. In fact, that ratio need not even
        be bounded, just not grow faster than PL does.
        """
        DpUPL = self.DpU * self.PL
        self.L = np.sum(DpUPL, axis=0) / self.I
        self.G = self.PL.T.dot(DpUPL) / self.I

    def profits(self, p: NPArrayType) -> NPArrayType:  # expects F elements
        self.utilities(p)
        self.probabilities()
        self.m = p - self.c
        for f in range(self.F):
            fi = self.Fis[f]
            self.pr[f] = self.P[fi].dot(self.m[fi])
        return self.pr

    def combgrad(self, p: NPArrayType) -> NPArrayType:  # expects J elements
        self._iterprep(p)
        cg = self.L * self.m + self.P
        for f in range(self.F):
            fi = self.Fis[f]
            for j in fi:
                for k in fi:
                    cg[j] -= self.G[k, j] * self.m[k]
            # G = np.array( self.G[fi,fi].T )
            # cg[fi] -= (self.G[fi,fi].T).dot( self.m[fi] )
        return cg

    def combgradz(self, p: NPArrayType) -> NPArrayType:  # expects J elements
        self.zeta_c(p)  # includes calling _iterprep
        return self.L * self.phi

    def probcheck(self, p: NPArrayType) -> None:

        print("probcheck: ")

        self.utilities(p)
        self.probabilities()
        P = np.array(self.P)  # force a copy, not reference

        self.lamgam()
        DP = -np.array(self.G)  # force copy
        for j in range(self.J):
            DP[j, j] += self.L[j]

        df, dh = np.zeros((self.J, self.J)), np.zeros(10)
        for h in range(10):
            H = 10 ** (-h)
            for j in range(self.J):
                p[j] += H
                self.utilities(p)
                self.probabilities()
                df[:, j] = (self.P - P) / H
                p[j] -= H
            dh[h] = np.abs(df - DP).max()
            print("  %0.8e: %0.2f %0.10f" % (H, np.log10(dh[h]), dh[h]))

    def gradcheck(self, p: NPArrayType) -> None:

        print("gradcheck: ")

        pr = np.array(self.profits(p))  # force a copy, not reference
        cg = self.combgrad(p)
        print(f"  cg - cz: { np.abs( cg - self.combgradz(p) ).max() }")

        df, dh = np.zeros(self.J), np.zeros(10)
        for h in range(10):
            H = 10 ** (-h)
            for f in range(self.F):
                fi = self.Fis[f]
                for j in self.Fis[f]:
                    p[j] += H
                    self.utilities(p)
                    self.probabilities()
                    prp = self.P[fi].dot(p[fi] - self.c[fi])
                    df[j] = (prp - pr[f]) / H
                    p[j] -= H
            dh[h] = np.abs(df - cg).max()
            print("  %0.8e: %0.2f %0.10f" % (H, np.log10(dh[h]), dh[h]))
