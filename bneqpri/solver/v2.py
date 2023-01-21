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
from time import time
from typing import List, Optional, Tuple, Union

import numpy as np

from ..types import NPArrayType
from ..utilities import Utility


class FPISolverV2:
    def __init__(
        self,
        products: int,  # number of products
        firms: int,  # number of firms
        products_per_firm: NPArrayType,  # number of products per firm, expects F elements
        costs: NPArrayType,  # expects J elements
        utility: Utility,
    ) -> None:
        r"""
        Solve for simultaneously stationary prices using fixed-point iteration (FPI)

        Args:
            products (int): total number of products
            firms (int): total number of firms
            products_per_firm (Array[int]): how many products each firm offers (length = firms)
                This presumes firm-sorted indexing for prices; that is

                    p[:products_per_firm[0]] -> prices for firm "0" (in any order)
                    p[products_per_firm[0]:products_per_firm[1]] -> prices for firm "1" (in any order)
                    ...

            costs (Array[int]): product costs (length = products)
            utility (Utility): a callable utility function; see imported ..utilities

        """

        # problem data
        self.J = products  # number of products
        self.F = firms  # number of firms
        self.Jf = products_per_firm  # list of the number of products per firm
        self.c = costs  # (unit) costs for each product
        self.utility = utility

        # internal data

        # indices segmenting firm blocks
        self.Fis: List[List[int]] = [[] for f in range(self.F)]
        self.Fis[0] = list(range(0, self.Jf[0]))
        for f in range(1, self.F):
            self.Fis[f] = list(
                range(self.Fis[f - 1][-1] + 1, self.Fis[f - 1][-1] + 1 + self.Jf[f])
            )

        # other storage
        self.m = np.zeros(self.J)  # markups (prices minus costs)
        self.pr = np.zeros(self.F)  # profits (convenience only)
        self.P = np.zeros(self.J)  # mixed probabilities
        self.L = np.zeros(self.J)  # "Lambda" operator, diagonal matrix
        self.G = np.zeros((self.J, self.J))  # "Gamma" operator, dense matrix
        self.z = np.zeros(self.J)  # "zeta" map
        self.phi = np.zeros(self.J)  # "phi" map, what we want to zero

        self.stats: List[List[Union[int, float]]] = []  # solve iteration "statistics"

    def solve(
        self,
        samples: int,  # _required_ (unless fixed sampling?)
        initial_prices: Optional[NPArrayType] = None,
        tolerance: float = 1.0e-6,
        max_iter: int = 1000,
        corrected: bool = True,
        batched: bool = False,
        memory: int = 10,
        verbose: bool = False,
        check: bool = False,
    ) -> NPArrayType:

        # for (annoying but maybe usefuul) consistency with math notation
        I = samples  # noqa: E741

        # random prices in [ c/2 , 3/2c ] if not specified
        p = (
            initial_prices
            if initial_prices is not None
            else (self.c / 2.0 + 2.0 * np.random.random(self.J))
        )

        # sample parameters needed to compute
        self.utility.sample(I)
        self.PL = np.zeros((I, self.J))  # logit probabilities (idiosyncratic)
        self.DpUPL = np.zeros((I, self.J))  # for Lambda/Gamma compute

        # define reference to corrected or uncorrected step
        _corrected = corrected and (self.utility.b is not None)
        step = self.zeta_c if _corrected else self.zeta_u

        self.max_iter = max_iter
        self.nrms = np.zeros((max_iter, 2))
        self.solved = False
        self.stats = []
        start = time()
        for self.iter in range(self.max_iter):

            if check:
                self.probcheck(p)
                self.gradcheck(p)

            # compute "step", ie the (corrected?) zeta map
            step(p, verbose=verbose)

            # test convergence (using step, not combined gradient)
            self.nrms[self.iter, 0] = np.max(np.abs(self.phi))
            self.nrms[self.iter, 1] = np.max(np.abs(self.L * self.phi))

            self.stats.append(
                [
                    self.iter,
                    time() - start,
                    p.min(),
                    p.max(),
                    self.pr.min(),
                    self.pr.max(),
                    self.P.sum(),
                    self.nrms[self.iter, 0],
                    self.nrms[self.iter, 1],
                ]
            )

            if verbose:
                self.profits(p)
                self._progress(p)

            if self.nrms[self.iter, 0] <= tolerance:
                self.solved = True
                break

            # fixed-point step, equivalently p -> p - phi = p - ( p - c - z )
            p = self.c + self.z

        self.time = time() - start

        self.nrms = self.nrms[: self.iter + 1, :]

        return p

    def summary(self) -> None:
        if self.solved:
            print(f"Solved in {self.iter}/{self.max_iter} steps, {self.time} seconds")
            print(f"fixed-point satisfaction |p-c-z| = {self.nrms[-1, 0]}")
            print(f"final combined gradient norm = {self.nrms[-1, 1]}")
        else:
            print(f"Failed to solve in {self.max_iter} steps, {self.time} seconds.")

    def _progress(self, p: NPArrayType) -> None:
        print(", ".join([str(s) for s in self.stats[-1]]))  # type: ignore

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

        maxbi, maxb = self.utility.maxbi, self.utility.maxb  # as sampled

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
            prFmi = self.PL[maxbi, fi].T.dot(self.m[fi])
            for j in fi:
                if p[j] > maxb:  # correction term - price j is too high
                    corr_idx.append((f, j))
                    # print(
                    #     f"WARNING ({self.iter}) -- {p[j]} > {self.maxb} correcting zeta for product {j}"
                    # )
                    self.z[j] = self.utility.DppU[maxbi, j] * (p[j] - maxb) + prFmi
                elif self.L[j] >= -1.0e-20:
                    # L[j] <= 0, so L[j] ~ 0.0, i.e. PL ~ 0.0
                    corr_idx.append((f, j))
                    # print(
                    #     f"WARNING ({self.iter}) -- p[{j}] = {p[j]} < {maxb} = maxinc but LAMp[{j}] = {self.L[j]}."
                    # )
                    # use a modification of extended map instead of what is calculated above
                    # z[j] = PL[maxinci,{f}]' * m[{f}]
                    self.z[j] = prFmi
                    # we exclude the "DppU[I*j+maxinci] * ( p[j] - maxinc )" term expecting
                    # p[j] to be at least close to maxinc
                else:
                    # normal, uncorrected
                    self.z[j] /= self.L[j]

        if verbose:
            print(f"Corrected: {corr_idx}")

        # compute phi = p - c - z also (self.m updated with _iterprep)
        self.phi = self.m - self.z

    def _iterprep(self, p: NPArrayType) -> None:
        self.probabilities(p)
        self.lamgam()
        self.m = p - self.c

    def _zeta_b(
        self,
    ) -> None:
        """z <- \tilde{GAMp}' * m - P"""
        for f in range(self.F):
            fi = self.Fis[f]
            self.pr[f] = self.P[fi].dot(self.m[fi])  # for stats only?
            # self.z[fi] = self.G[fi,fi].T.dot(self.m[fi]) - self.P[fi]
            self.z[fi] = -self.P[fi]
            for j in fi:
                for k in fi:
                    self.z[j] += self.G[k, j] * self.m[k]

    def probabilities(self, p: NPArrayType) -> None:

        self.utility(p)  # this is the only place we compute this

        I = self.PL.shape[0]  # noqa: E741
        self.PL, self.P = 0.0 * self.PL, 0.0 * self.P  # zero out, not reallocate

        uimax = np.maximum(0, np.max(self.utility.U, axis=1))  # exp float fix

        # TODO: unbounded loop, figure out broadcast mechanism?
        for i in range(I):
            self.PL[i, :] = np.exp(self.utility.U[i, :] - uimax[i])
            self.PL[i, :] /= np.exp(-uimax[i]) + self.PL[i, :].sum()

        # # TODO: bounded loop, but double loop, better approach?
        # for j in range(self.J):
        #     self.PL[:, j] = np.exp(self.utility.U[:, j] - uimax[:])
        # S = np.exp(-uimax) + np.sum(self.PL, axis=1) # I vector
        # for j in range(self.J):
        #     self.PL[:, j] /= S[:]

        self.P = np.sum(self.PL, axis=0) / I

    def lamgam(self) -> None:
        """Compute the "Lambda" and "Gamma" matrices from the papers. Note
        that "Lambda" (`L`) is a diagonal matrix, but "Gamma" (`G`) is full.

        Also note some theory:

            DpU * PL -> 0 as p -> b

        so long as

              DpPL -> 0 and - DppU/(DpU)^2 is bounded as p -> b

        as follows from L'Hopital's rule. Moreover, this is sufficient but
        not necessary, though boundedness of that ratio of derivatives is
        a totally reasonable ask.
        """
        # TODO: in-place update or allocation?
        self.DpUPL[:, :] = self.utility.DpU[:, :] * self.PL[:, :]
        self.L = np.sum(self.DpUPL, axis=0) / self.PL.shape[0]
        self.G = self.PL.T.dot(self.DpUPL) / self.PL.shape[0]

    def profits(self, p: NPArrayType) -> NPArrayType:  # expects F elements
        self.probabilities(p)
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

        self.probabilities(p)
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
                self.probabilities(p)
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
                    self.probabilities(p)
                    prp = self.P[fi].dot(p[fi] - self.c[fi])
                    df[j] = (prp - pr[f]) / H
                    p[j] -= H
            dh[h] = np.abs(df - cg).max()
            print("  %0.8e: %0.2f %0.10f" % (H, np.log10(dh[h]), dh[h]))


# TBD: naive batching does _not_ work (yet)

# class BatchedFPISolverV2(FPISolverV2):
#     def solve(
#         self,
#         samples: int,  # _required_ (unless fixed sampling?)
#         initial_prices: Optional[NPArrayType] = None,
#         tolerance: float = 1.0e-6,
#         max_iter: int = 1000,
#         corrected: bool = True,
#         memory: int = 10,  # violates liskov sub principle
#         verbose: bool = False,
#         check: bool = False,
#     ) -> NPArrayType:

#         # for (annoying but maybe usefuul) consistency with math notation
#         I = samples  # noqa: E741

#         # random prices in [ c/2 , 3/2c ] if not specified
#         p = (
#             initial_prices
#             if initial_prices is not None
#             else ((0.5 + np.random.random(self.J)) * self.c)
#         )

#         # sample parameters needed to compute
#         self.PL = np.zeros((I, self.J))  # logit probabilities (idiosyncratic)
#         self.DpUPL = np.zeros((I, self.J))  # for Lambda/Gamma compute

#         # define reference to corrected or uncorrected step
#         _corrected = corrected and (self.utility.b is not None)
#         step = self.zeta_c if _corrected else self.zeta_u

#         Zs = np.zeros((self.J, memory))
#         Ws = np.zeros(memory)
#         _m = 0

#         self.max_iter = max_iter
#         self.nrms = np.zeros((max_iter, 2))
#         self.solved = False
#         self.stats = []
#         start = time()
#         for self.iter in range(self.max_iter):

#             if check:
#                 self.probcheck(p)
#                 self.gradcheck(p)

#             self.utility.sample(I)

#             # compute "step", ie the (corrected?) zeta map
#             step(p, verbose=verbose)

#             # test convergence (using step, not combined gradient)
#             self.nrms[self.iter, 0] = np.max(np.abs(self.phi))
#             self.nrms[self.iter, 1] = np.max(np.abs(self.L * self.phi))

#             self.stats.append(
#                 [
#                     self.iter,
#                     time() - start,
#                     p.min(),
#                     p.max(),
#                     self.pr.min(),
#                     self.pr.max(),
#                     self.P.sum(),
#                     self.nrms[self.iter, 0],
#                     self.nrms[self.iter, 1],
#                 ]
#             )

#             if verbose:
#                 self.profits(p)
#                 self._progress(p)

#             if self.nrms[self.iter, 0] <= tolerance:
#                 self.solved = True
#                 break

#             # fixed-point step, equivalently p -> p - phi = p - ( p - c - z )
#             # but batched with memory
#             Ws = Ws - 1  # decrement existing weights
#             Ws[_m] = memory  # set weight of new entry
#             Zs[:, _m] = self.z[:]  # highest weight sample is current
#             _m = 0 if _m == memory - 1 else (_m + 1)
#             s = Zs @ (Ws / Ws.sum())
#             print(Ws, s, self.z)
#             p = self.c + s

#         self.time = time() - start

#         self.nrms = self.nrms[: self.iter + 1, :]

#         return p
