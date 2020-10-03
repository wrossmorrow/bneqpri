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

import argparse

header = """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Bertrand-Nash Equilibrium Prices Solver (with Finite Purchasing Power / Budgets)

Method from: 

Morrow, W.R. & Skerlos, S.J. Fixed-Point Approaches to Computing Bertrand-Nash Equilibrium 
Prices Under Mixed-Logit Demand. Operations Research 59(2) (2011).
https://doi.org/10.1287/opre.1100.0894)

Morrow, W.R. Finite purchasing power and computations of Bertrand–Nash equilibrium prices.
Computational Optimization and Applications 62, 477–515 (2015). 
https://doi.org/10.1007/s10589-015-9743-7           

Note this software is provided AS-IS under the GPL v2.0 License. Contact the author
with any questions. Copyright 2020+ W. Ross Morrow. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
"""

def cli():

    parser = argparse.ArgumentParser(description=header, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--format', 
        type=str, 
        choices=["sequential","indexed"],
        default="sequential",
        help="""Format input files are written in. Sequential presumes firms' products 
are listed sequentially, indexed means the header for product and individual file has _
indices describing the firm and product for the column. NOT YET IMPLEMENTED."""
    )

    parser.add_argument(
        '--firms', 
        type=str, 
        help='File describing the number of firms and the number of products they each offer'
    )

    parser.add_argument(
        '--products', 
        type=str, 
        help='File describing the number of products and their unit costs'
    )

    parser.add_argument(
        '--individuals', 
        type=str, 
        help='File describing the number of individuals, their budgets, price sensitivities, and utilities'
    )

    parser.add_argument(
        '--linear-utility', 
        dest='linear', 
        action='store_true', 
        default=False,
        help='Presume the utility is linear in the price coefficient, instead of using budgets'
    )

    parser.add_argument(
        '--initial-prices', 
        type=str, 
        help='File with initial prices to use when starting iterations'
    )

    parser.add_argument(
        '--prices', 
        type=str, 
        default="prices.csv",
        help='File with the computed prices, written upon solve'
    )

    parser.add_argument(
        '--ftol', 
        type=float, 
        default=1.0e-6,
        help='Solve tolerance, will terminate when |p-c-z(p)| < ftol'
    )

    parser.add_argument(
        '--iters', 
        type=int, 
        default=1000,
        help='Maximum number of iterations'
    )

    parser.add_argument(
        '--no-header', 
        dest='header',
        action='store_false', 
        default=False,
        help="Don't print header about the code/method"
    )

    parser.add_argument(
        '-q', '--quiet', 
        dest='verbose',
        action='store_false', 
        default=True,
        help="Don't print information to the console"
    )

    return parser.parse_args()

