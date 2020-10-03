#!/bin/bash

for DIR in examples/linear/*/ ; do 
	python -m bneqpri \
		--no-header \
		--linear-utility \
		--firms ${DIR}firms.csv \
		--products ${DIR}products.csv \
		--individuals ${DIR}utilities.csv \
		--prices ${DIR}prices.csv \
		--ftol 1.0e-8
done

for DIR in examples/budgets/*/ ; do
	python -m bneqpri \
		--no-header \
		--firms ${DIR}firms.csv \
		--products ${DIR}products.csv \
		--individuals ${DIR}utilities.csv \
		--prices ${DIR}prices.csv \
		--ftol 1.0e-8
done
