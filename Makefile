# Directory for the virtual environment
VENV_DIR = venv

# Python executable
PYTHON = ${VENV_DIR}/bin/python

# Pip executable
PIP = ${VENV_DIR}/bin/pip

all:
	@echo "Usage:"
	@echo "  make venv - create a virtual environment"
	@echo "  make install - execute venv and install dependencies"
	@echo "  make clean - remove the virtual environment and compiled files"
	@echo "\nTo activate the virtual environment, run 'source venv/bin/activate'"

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
	fi

install: venv
	${PIP} install -r requirements.txt

clean:
	rm -rf $(VENV_DIR)
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

.PHONY: all clean install venv
