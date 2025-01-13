#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ft_dslr
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

VENV_DIR = .venv
PYTHON = ${VENV_DIR}/bin/python
PIP = ${VENV_DIR}/bin/pip
DATA_URL = https://cdn.intra.42.fr/document/document/29579/datasets.tgz

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## basic info
all:
	@echo "Usage:"
	@echo "  make ${VENV_DIR} - create a virtual environment"
	@echo "  make install - execute venv and install dependencies"
	@echo "  make clean - remove the virtual environment and compiled files"
	@echo "\nTo activate the virtual environment, run 'source venv/bin/activate'"

## Create the python virtual env
$(VENV_DIR):
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
	fi

## Install the requirements.txt
.PHONY: install
install: requirements.txt  $(VENV_DIR)
	${PIP} install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ${PROJECT_NAME} --disable-noqa --config setup.cfg
	isort --check --diff --profile black ${PROJECT_NAME}
	black --check --config pyproject.toml ${PROJECT_NAME}

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml ${PROJECT_NAME}
	isort ${PROJECT_NAME} --settings-path pyproject.toml


## Download required data from the 42 intra
.PHONY: data
data:
	wget ${DATA_URL}
	[ -d ./data/external ] || mkdir -p ./data/external
	tar -xvf datasets.tgz -C ./data/external/
	rm -rf datasets.tgz



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
