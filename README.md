# eqpri

Python code for computing equilibrium prices

# Quick Start

Run by specifying three `csv` files, describing the firm structure, the products (costs only now), and the utilities. The firm file should be a single-line `csv` file (not including header) with the number of firms `F`, followed by the number of products offered by each firm. E.g, 
```
3 , 3 , 5 , 4
```
Spaces are ignored. The firm file thus has `F+1` columns. The product file is another single-line `csv` file (not including header) and should contain the number of products `J`, followed by the costs for each of the `J` products. `J` must equal the sum of the number of products offered by each firm in the firm file. The products file thus has `J+1` columns. The utilities file is the most complicated, having as many rows as the number of individuals `I` (not including header) and `2+J` columns. In each row `i`, which corresponds to an individual `i`, the first column should be the budget for that individual, the second the price sensitivity, and the remaining `J` columns have the total non-price utility of all the part-worths of the features for each product in columns `2` for product 1, `2+1` for product 2, etc. 

Example files are provided in the `example` directory here. 

With such files, run as follows: 

```
$ python fppep.py example/firms.csv example/products.csv example/utilities.csv prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Bertrand-Nash Equilibrium Prices with Budgets (Finite Purchasing Power)

Method from: 

Morrow, W.R. Finite purchasing power and computations of Bertrand–Nash equilibrium prices. 
Comput Optim Appl 62, 477–515 (2015). https://doi.org/10.1007/s10589-015-9743-7

Note this software is provided AS-IS under the GPL v2.0 License. Contact the author
with any questions. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2020-09-29T22:07:57.901019 :: Preparing data
2020-09-29T22:07:57.901097 :: Starting solve
2020-09-29T22:07:58.595876 :: Solved in 14/1000 steps, 0.6947450637817383 seconds, |p-c-z| = 2.27665303687008e-06
2020-09-29T22:07:58.596230 :: (probable) equilibium prices written to prices.csv

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

# Details

Detailed README content TBD

# TBD

* Tons more tests
* Implement `argparse` for good options
* Figure out why `numpy` syntax isn't computing `G'm` terms correctly
* Turn into a proper package

# Contact

[W. Ross Morrow](morrowwr@gmail.com)

Copyright 2020+
