# bneqpri

Python code for computing Bertrand-Nash equilibrium prices with "mixed" logit models. The implemented method is the fixed-point iteration derived and discussed in the papers [Fixed-Point Approaches to Computing Bertrand-Nash Equilibrium Prices Under Mixed-Logit Demand](https://doi.org/10.1287/opre.1100.0894) (_Operations Research_, 2011) and [Finite purchasing power and computations of Bertrand–Nash equilibrium prices](https://doi.org/10.1007/s10589-015-9743-7) (_Computational Optimization and Applications_, 2015); there's also a super verbose working paper on the [arxiv](https://arxiv.org/abs/1012.5836). You can get the whole gist from the first paper, the second introduces finer grained details related to treatment of "budgets" in the choice model (under regularity conditions). The basic idea is that there is a _particular_ fixed-point equation for simultaneous stationarity (the _necessary_ conditions for equilibrium prices) that is provably norm-coercive and generates steps that are never orthogonal to the combined-gradient. Iterating such steps appears to be a strong alternative to Newton-type methods as discussed at some length in the papers, although there is no convergence proof. 

# Dependencies

Using this repo requires `python3` and `numpy`. 

# Installation

This project is on [pypi](https://pypi.org/project/bneqpri/) and thus you can use `pip`: 
```shell
pip install bneqpri
```

If you want to build from the repo, do 
```shell
just build
```
if you have [`just`](https://github.com/casey/just). Or
```shell
poetry build
```
with [`poetry`](https://python-poetry.org/). 

You can run tests with 
```shell
just unit-test
```
or review the standard-ish `pytest` command in the `justfile`. 

# Command-Line Usage

```
$ python -m bneqpri --help
usage: __main__.py [-h] [--format {sequential,indexed}] [--firms FIRMS]
                   [--products PRODUCTS] [--individuals INDIVIDUALS]
                   [--linear-utility] [--initial-prices INITIAL_PRICES]
                   [--prices PRICES] [--ftol FTOL] [--iters ITERS] [-q]

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
  --linear-utility      Presume the utility is linear in the price coefficient, instead of using budgets
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

## Models with Finite Purchasing Power / Budgets

The utilities file is the most complicated, having as many rows as the number of individuals `I` (not including header) and `2+J` columns. In each row `i`, which corresponds to an individual `i`, the first column should be the budget for that individual `b[i]`, the second the price sensitivity `a[i]`, and the remaining `J` columns have the total non-price utility of all the part-worths of the features for each product in columns `2` for product 1, `2+1` for product 2, etc comprising a matrix `V[i,j]` of product utilities (except price). Again, products are ordered sequentially by firms. 

The utility function for choices used here is, for now, presumed to be 
```
U[i,j](p) = a[i] log( b[i] - p ) + V[i,j]
```
when there is a budget. There is an outside good with unit value; any non unit value should be absorbed into the non-price utility. A word of warning, the price sensitivity _must be_ positive for all individuals and _probably_ needs to be larger than one for all individuals. 

## Linear Utility

We also include an implementation and command line flag (`--linear-utility`) for linear utility models 
```
U[i,j](p) = a[i] p + V[i,j]
```
Note that here `a[i]` should be negative and you should not specify the `b[i]` column in the utilities file (thus it will have only `1+J` columns, not `2+J`). 


# Examples

Example files of each are provided in the `example` directory here, and you can run them all as follows: 

```
$ bash examples.sh 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:26:37.144882 :: Modeling 1000 individuals, 10 firms, 144 products, linear utility no budget
2020-10-02T20:26:37.687914 :: Solved in 34/1000 steps, 0.542978048324585 seconds
2020-10-02T20:26:37.687948 :: fixed-point satisfaction |p-c-z| = 7.961977299686396e-09
2020-10-02T20:26:37.688401 :: (probable) equilibium prices written to examples/linear/1000-10-144/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:26:37.918779 :: Modeling 1000 individuals, 5 firms, 79 products, linear utility no budget
2020-10-02T20:26:38.287736 :: Solved in 28/1000 steps, 0.36890220642089844 seconds
2020-10-02T20:26:38.287766 :: fixed-point satisfaction |p-c-z| = 8.772539583645766e-09
2020-10-02T20:26:38.288218 :: (probable) equilibium prices written to examples/linear/1000-5-79/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:26:38.465683 :: Modeling 50 individuals, 3 firms, 53 products, linear utility no budget
2020-10-02T20:26:38.532921 :: Solved in 44/1000 steps, 0.06718897819519043 seconds
2020-10-02T20:26:38.532958 :: fixed-point satisfaction |p-c-z| = 6.891536230568818e-09
2020-10-02T20:26:38.533299 :: (probable) equilibium prices written to examples/linear/50-3-53/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:26:38.730063 :: Modeling 500 individuals, 3 firms, 40 products, linear utility no budget
2020-10-02T20:26:39.188571 :: Solved in 78/1000 steps, 0.4584488868713379 seconds
2020-10-02T20:26:39.188606 :: fixed-point satisfaction |p-c-z| = 8.38127345303974e-09
2020-10-02T20:26:39.188918 :: (probable) equilibium prices written to examples/linear/500-3-40/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:26:39.457582 :: Modeling 1000 individuals, 10 firms, 148 products, nonlinear utility with budget
2020-10-02T20:26:51.156455 :: Solved in 32/1000 steps, 11.698816061019897 seconds
2020-10-02T20:26:51.156493 :: fixed-point satisfaction |p-c-z| = 8.289176345321891e-09
2020-10-02T20:26:51.156979 :: (probable) equilibium prices written to examples/budgets/1000-10-148/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:26:51.393449 :: Modeling 1000 individuals, 5 firms, 68 products, nonlinear utility with budget
2020-10-02T20:27:03.496671 :: Solved in 73/1000 steps, 12.103170156478882 seconds
2020-10-02T20:27:03.496717 :: fixed-point satisfaction |p-c-z| = 9.146744250898564e-09
2020-10-02T20:27:03.497178 :: (probable) equilibium prices written to examples/budgets/1000-5-68/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:27:03.689047 :: Modeling 50 individuals, 3 firms, 41 products, nonlinear utility with budget
2020-10-02T20:27:03.818524 :: Solved in 20/1000 steps, 0.12943220138549805 seconds
2020-10-02T20:27:03.818557 :: fixed-point satisfaction |p-c-z| = 5.774188682750037e-09
2020-10-02T20:27:03.818899 :: (probable) equilibium prices written to examples/budgets/50-3-41/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:27:04.019609 :: Modeling 500 individuals, 3 firms, 44 products, nonlinear utility with budget
2020-10-02T20:27:08.500028 :: Solved in 78/1000 steps, 4.48035192489624 seconds
2020-10-02T20:27:08.500067 :: fixed-point satisfaction |p-c-z| = 8.66341531846615e-09
2020-10-02T20:27:08.500400 :: (probable) equilibium prices written to examples/budgets/500-3-44/prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

# Details

Detailed README content TBD

# TBD

* Tons more tests
* Implement second-order sufficiency check for computed prices
* Figure out why `numpy` syntax isn't computing `G'm` terms correctly

# Contact

[W. Ross Morrow](morrowwr@gmail.com)

Copyright 2020+
