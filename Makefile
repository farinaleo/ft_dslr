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

all: ## basic info
	@echo "Usage:"
	@echo "  make ${VENV_DIR} - create a virtual environment"
	@echo "  make install - execute venv and install dependencies"
	@echo "  make clean - remove the virtual environment and compiled files"
	@echo "\nTo activate the virtual environment, run 'source venv/bin/activate'"


$(VENV_DIR): ## Create the python virtual env
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
	fi


.PHONY: install
install: requirements.txt  $(VENV_DIR) ## Install the requirements.txt
	${PIP} install -r requirements.txt


.PHONY: clean
clean: ## Delete all compiled Python files
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


.PHONY: lint
lint: ## Lint using flake8 and black
	flake8 ${PROJECT_NAME} --disable-noqa --config setup.cfg
	isort --check --diff --profile black ${PROJECT_NAME}
	black --check --config pyproject.toml ${PROJECT_NAME}


.PHONY: format
format: ## Format source code with black
	black --config pyproject.toml ${PROJECT_NAME}
	isort ${PROJECT_NAME} --profile black


.PHONY: data
data: ## Download required data from the 42 intra
	wget ${DATA_URL}
	[ -d ./data/external ] || mkdir -p ./data/external
	tar -xvf datasets.tgz -C ./data/external/
	rm -rf datasets.tgz


.PHONY: build_docs
build_docs: ## Build the html documentation
	pydoctor \
    --project-name=${PROJECT_NAME} \
    --project-version=0.1 \
    --project-url=https://github.com/farinaleo/${PROJECT_NAME}/ \
	--html-base-url=https://farinaleo.github.io/${PROJECT_NAME}/ \
    --docformat=restructuredtext \
    --theme=readthedocs \
    ./ft_dslr


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help:
	@echo Available rules:
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'
