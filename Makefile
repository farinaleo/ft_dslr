# Directory for the virtual environment
VENV_DIR = .venv

# Python executable
PYTHON = ${VENV_DIR}/bin/python

# Pip executable
PIP = ${VENV_DIR}/bin/pip

all:
	@echo "Usage:"
	@echo "  make ${VENV_DIR} - create a virtual environment"
	@echo "  make install - execute venv and install dependencies"
	@echo "  make clean - remove the virtual environment and compiled files"
	@echo "\nTo activate the virtual environment, run 'source venv/bin/activate'"

$(VENV_DIR):
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
	fi

install: $(VENV_DIR)
	${PIP} install -r requirements.txt

clean:
	rm -rf $(VENV_DIR)
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

lint:
	flake8 srcs  --disable-noqa --config setup.cfg
	isort --check --diff --profile black srcs
	black --check --config pyproject.toml srcs

format:
	black srcs --config pyproject.toml
	isort srcs --settings-path pyproject.toml


.PHONY: all clean install $(VENV_DIR)) lint format
