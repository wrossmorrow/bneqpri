# eqpri

Python code for computing equilibrium prices

# Dependencies

This requires `python3` and `numpy`. 

# Command-Line Usage

```
(base) mememe:eqpri morrowwr$ python fppep.py --help
usage: fppep.py [-h] [--format {sequential,indexed}] [--firms FIRMS]
                [--products PRODUCTS] [--individuals INDIVIDUALS]
                [--initial-prices INITIAL_PRICES] [--prices PRICES]
                [--ftol FTOL] [--iters ITERS] [-q]

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Bertrand-Nash Equilibrium Prices with Budgets (Finite Purchasing Power)

Method from: 

  Morrow, W.R. Finite purchasing power and computations of Bertrand–Nash equilibrium prices. 
  Comput Optim Appl 62, 477–515 (2015). https://doi.org/10.1007/s10589-015-9743-7

Note this software is provided AS-IS under the GPL v2.0 License. Contact the author
with any questions. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

optional arguments:
  -h, --help            show this help message and exit
  --format {sequential,indexed}
                        Format input files are written in. Sequential presumes firms' products 
                        are listed sequentially, indexed means the header for product and individual file has _
                        indices describing the firm and product for the column. NOT YET IMPLEMENTED.
  --firms FIRMS         File describing the number of firms and the number of products they each offer
  --products PRODUCTS   File describing the number of products and their unit costs
  --individuals INDIVIDUALS
                        File describing the number of individuals, their budgets, price sensitivities, and utilities
  --initial-prices INITIAL_PRICES
                        File with initial prices to use when starting iterations
  --prices PRICES       File with the computed prices, written upon solve
  --ftol FTOL           Solve tolerance, will terminate when |p-c-z(p)| < ftol
  --iters ITERS         Maximum number of iterations
  -q, --quiet           Don't print information to the console
```

# Quick Start

Run by specifying three `csv` files, describing the firm structure, the products (costs only now), and the utilities. 

The firm file should be a single-line `csv` file (not including header) with the number of firms `F`, followed by the number of products offered by each firm. E.g, 
```
3 , 3 , 5 , 4
```
for 3 firms, the first of which offers 3 products, the second 5, and the third 4. Spaces are ignored. The firm file thus has `F+1` columns. 

The product file is another single-line `csv` file (not including header) and should contain the number of products `J`, followed by the costs for each of the `J` products. `J` must equal the sum of the number of products offered by each firm in the firm file. E.g., 
```
12 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 3 , 3 , 3
```
The products file thus has `J+1` columns. **Note** that this code presumes the products are ordered by firm. In our example above, we've chosen unit costs such that firm 1's 3 products all cost "1" to make, firm 2's "2", and firm 3's "3". Note that the monetary units are irrelevant as long as they are consistent with utilities and budgets, and prices are computed in whatever units costs are in. 

The utilities file is the most complicated, having as many rows as the number of individuals `I` (not including header) and `2+J` columns. In each row `i`, which corresponds to an individual `i`, the first column should be the budget for that individual `b[i]`, the second the price sensitivity `a[i]`, and the remaining `J` columns have the total non-price utility of all the part-worths of the features for each product in columns `2` for product 1, `2+1` for product 2, etc comprising a matrix `V[i,j]` of product utilities (except price). Again, products are ordered sequentially by firms. 

The utility function for choices used here is, for now, presumed to be 
```
U[i,j](p) = a[i] log( b[i] - p ) + V[i,j]
```
There is an outside good with unit value; any non unit value should be absorbed into the non-price utility. A word of warning, the price sensitivity _must be_ positive for all individuals and _probably_ needs to be larger than one for all individuals. 

Example files of each are provided in the `example` directory here. 

With such files, run the code as follows: 

```
$ python fppep.py example/firms.csv example/products.csv example/utilities.csv prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Bertrand-Nash Equilibrium Prices with Budgets (Finite Purchasing Power)

Method from: 

( Morrow, W.R. Finite purchasing power and computations of Bertrand–Nash equilibrium prices. )
( Comput Optim Appl 62, 477–515 (2015). https://doi.org/10.1007/s10589-015-9743-7            )

Currently only uses the utility function 

U[i,j] = a[i] log( b[i] - p[j] ) + V[i,j]

where a[i] is individual i's price sensitivity, b[i] is their budget, p[j] is the 
price of product j, and V[i,j] is the total of all non-price components of utility.

Note this software is provided AS-IS under the GPL v2.0 License. Contact the author
with any questions. Copyright 2020+ W. Ross Morrow. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-09-30T07:10:25.817446 :: Modeling 500 individuals, 3 firms, 37 products
2020-09-30T07:10:25.864241 :: Solved in 1/1000 steps, 0.046675920486450195 seconds
2020-09-30T07:10:25.864274 :: fixed-point satisfaction |p-c-z| = 6.8332302038953685e-09
2020-09-30T07:10:25.864576 :: (probable) equilibium prices written to new-prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

# Details

Detailed README content TBD

# TBD

* Tons more tests
* Implement second-order sufficiency check for computed prices
* Figure out why `numpy` syntax isn't computing `G'm` terms correctly
* Turn into a proper package

# Contact

[W. Ross Morrow](morrowwr@gmail.com)

Copyright 2020+
