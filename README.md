# bneqpri

Python code for computing Bertrand-Nash equilibrium prices with "mixed" logit models. The implemented method is the fixed-point iteration derived and discussed in the papers [Fixed-Point Approaches to Computing Bertrand-Nash Equilibrium Prices Under Mixed-Logit Demand](https://doi.org/10.1287/opre.1100.0894) and [Finite purchasing power and computations of Bertrand–Nash equilibrium prices](https://doi.org/10.1007/s10589-015-9743-7). 

# Dependencies

This requires `python3` and `numpy`. 

# Installation

This project is on [pypi](https://pypi.org/project/bneqpri/) and thus you can use `pip`: 

```
$ pip install bneqpri
```

If you want to build from the repo, do 

```
$ python setup.py sdist bdist_wheel
$ pip install --find-links ./dist/
```

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

2020-10-02T20:09:19.011956 :: Modeling 1000 individuals, 10 firms, 144 products, linear utility no budget
2020-10-02T20:09:19.526557 :: Solved in 34/1000 steps, 0.5145268440246582 seconds
2020-10-02T20:09:19.526595 :: fixed-point satisfaction |p-c-z| = 7.961977299686396e-09
2020-10-02T20:09:19.527064 :: (probable) equilibium prices written to examples/linear/1000-10-144//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:19.752497 :: Modeling 1000 individuals, 5 firms, 79 products, linear utility no budget
2020-10-02T20:09:20.105496 :: Solved in 28/1000 steps, 0.3529479503631592 seconds
2020-10-02T20:09:20.105529 :: fixed-point satisfaction |p-c-z| = 8.772539583645766e-09
2020-10-02T20:09:20.105973 :: (probable) equilibium prices written to examples/linear/1000-5-79//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:20.279930 :: Modeling 50 individuals, 3 firms, 53 products, linear utility no budget
2020-10-02T20:09:20.342641 :: Solved in 44/1000 steps, 0.06260204315185547 seconds
2020-10-02T20:09:20.342679 :: fixed-point satisfaction |p-c-z| = 6.891536230568818e-09
2020-10-02T20:09:20.343049 :: (probable) equilibium prices written to examples/linear/50-3-53//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:20.534591 :: Modeling 500 individuals, 3 firms, 40 products, linear utility no budget
2020-10-02T20:09:20.989304 :: Solved in 78/1000 steps, 0.45466113090515137 seconds
2020-10-02T20:09:20.989340 :: fixed-point satisfaction |p-c-z| = 8.38127345303974e-09
2020-10-02T20:09:20.989679 :: (probable) equilibium prices written to examples/linear/500-3-40//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:21.249497 :: Modeling 1000 individuals, 10 firms, 148 products, nonlinear utility with budget
2020-10-02T20:09:32.021934 :: Solved in 32/1000 steps, 10.772387981414795 seconds
2020-10-02T20:09:32.021968 :: fixed-point satisfaction |p-c-z| = 8.289176345321891e-09
2020-10-02T20:09:32.022484 :: (probable) equilibium prices written to examples/budgets/1000-10-148//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:32.252172 :: Modeling 1000 individuals, 5 firms, 68 products, nonlinear utility with budget
2020-10-02T20:09:44.043457 :: Solved in 73/1000 steps, 11.791236162185669 seconds
2020-10-02T20:09:44.043489 :: fixed-point satisfaction |p-c-z| = 9.146744250898564e-09
2020-10-02T20:09:44.043839 :: (probable) equilibium prices written to examples/budgets/1000-5-68//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:44.237175 :: Modeling 50 individuals, 3 firms, 41 products, nonlinear utility with budget
2020-10-02T20:09:44.363257 :: Solved in 20/1000 steps, 0.12602996826171875 seconds
2020-10-02T20:09:44.363291 :: fixed-point satisfaction |p-c-z| = 5.774188682750037e-09
2020-10-02T20:09:44.363713 :: (probable) equilibium prices written to examples/budgets/50-3-41//prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-10-02T20:09:44.566969 :: Modeling 500 individuals, 3 firms, 44 products, nonlinear utility with budget
2020-10-02T20:09:48.822381 :: Solved in 78/1000 steps, 4.255356073379517 seconds
2020-10-02T20:09:48.822420 :: fixed-point satisfaction |p-c-z| = 8.66341531846615e-09
2020-10-02T20:09:48.822726 :: (probable) equilibium prices written to examples/budgets/500-3-44//prices.csv

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
