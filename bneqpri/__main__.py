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

from bneqpri.cli import cli , header
from bneqpri.io import read_data_files , read_initial_prices , write_prices
from bneqpri.logging import log
from bneqpri.impl import BUFPISolver , LUFPISolver

if __name__ == "__main__": 

    args = cli()

    if args.firms is None: 
        print( f"\nbneqpri requires a firms file (--firms)\n" )
        exit(1)

    if args.products is None: 
        print( f"\nbneqpri requires a products file (--products)\n" )
        exit(1)

    if args.individuals is None: 
        print( f"\nbneqpri requires an individuals file (--individuals)\n" )
        exit(1)

    if args.header: 
        print( header )
    elif args.verbose: 
        print( """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
""" )

    if args.linear: 
        I, J, F, Jf, a, V, c = read_data_files(args.firms, args.products, args.individuals)
        S = LUFPISolver(I, J, F, Jf, a, V, c)
        extra = ", linear utility no budget"
    else: 
        I, J, F, Jf, b, a, V, c = read_data_files(args.firms, args.products, args.individuals, budget=True)
        S = BUFPISolver(I, J, F, Jf, b, a, V, c)
        extra = ", nonlinear utility with budget"

    if args.verbose: 
        log( f"Modeling {S.I} individuals, {S.F} firms, {S.J} products{extra}" )

    if args.initial_prices is not None: 
        p0 = read_initial_prices( args.initial_prices )
    else: 
        p0 = c

    p = S.solve( p0=p0 , f_tol=args.ftol , max_iter=args.iters, verbose=False )

    if S.solved : 

        if args.verbose: 
            log( f"Solved in {S.iter}/{args.iters} steps, {S.time} seconds" )
            log( f"fixed-point satisfaction |p-c-z| = {S.nrms[-1]}" )
        write_prices( args.prices , p )
        if args.verbose: 
            log( f"(probable) equilibium prices written to {args.prices}" )

    else: 

        if args.verbose: 
            log( f"Failed to solve in {args.iters} steps, {S.time} seconds." )

    if args.verbose: 
        print( """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
""" )