default:
    just --list

name := "bneqpri"

alias i := install
alias u := update
alias f := format
alias l := lint
alias t := unit-test
alias b := build
# alias p := publish

# install dependencies
install: 
    poetry install

# update dependencies
update: 
    poetry update

# format the code
format: 
    poetry run black {{name}}
    poetry run black test

# run mypy static type analysis
types: 
    poetry run mypy {{name}}

# lint the code
lint:
    poetry run black {{name}} test --check --exclude env
    poetry run flake8 {{name}} test
    poetry run mypy {{name}} test
    
examples:
    poetry run bash ./examples.sh

# run all unit tests
unit-test *flags:
    poetry run python -m pytest -v \
        test/unit \
        --disable-warnings \
        {{flags}}

# run all solver tests
solver-test *flags:
    poetry run python -m pytest -v \
        test/solver \
        --disable-warnings \
        {{flags}}
            
# run all integration tests
integration-test *flags:
    poetry run python -m pytest -v \
        test/integration \
        --disable-warnings \
        {{flags}}

# build package
build: 
    poetry build

# publish the package
publish *flags:
    poetry publish {{flags}}
