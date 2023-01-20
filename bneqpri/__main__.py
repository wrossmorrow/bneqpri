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
from logging import getLogger

from .cli import cli, header
from .config import read_data_files, read_initial_prices, write_prices
from .solver import FPISolver
from .impl import (
    LinearUtilityFPISolver,
    LogOfRemainingUtilityFPISolver,
)

logger = getLogger(__name__)


if __name__ == "__main__":

    args = cli()

    if args.firms is None:
        logger.error("\nbneqpri requires a firms file (--firms)\n")
        exit(1)

    if args.products is None:
        logger.error("\nbneqpri requires a products file (--products)\n")
        exit(1)

    if args.individuals is None:
        logger.error("\nbneqpri requires an individuals file (--individuals)\n")
        exit(1)

    if args.header:
        print(header)
    elif args.verbose:
        print(
            """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
        )

    S: FPISolver
    if args.linear:
        I, J, F, Jf, a, V, c = read_data_files(args.firms, args.products, args.individuals)
        S = LinearUtilityFPISolver(I, J, F, Jf, a, V, c)
        extra = ", linear utility no budget"
    else:
        I, J, F, Jf, b, a, V, c = read_data_files(
            args.firms, args.products, args.individuals, budget=True
        )
        S = LogOfRemainingUtilityFPISolver(I, J, F, Jf, b, a, V, c)
        extra = ", log-of-remaining-utility with budget"

    if args.verbose:
        logger.info(f"Modeling {S.I} individuals, {S.F} firms, {S.J} products{extra}")

    if args.initial_prices is not None:
        p0 = read_initial_prices(args.initial_prices)
    else:
        p0 = c

    p = S.solve(p0=p0, f_tol=args.ftol, max_iter=args.iters, verbose=False)

    if S.solved:

        if args.verbose:
            logger.info(f"Solved in {S.iter}/{args.iters} steps, {S.time} seconds")
            logger.info(f"fixed-point satisfaction |p-c-z| = {S.nrms[-1]}")
        write_prices(args.prices, p)
        if args.verbose:
            logger.info(f"(probable) equilibium prices written to {args.prices}")

    else:

        if args.verbose:
            logger.error(f"Failed to solve in {args.iters} steps, {S.time} seconds.")

    if args.verbose:
        print(
            """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
        )
