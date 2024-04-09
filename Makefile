run: venv/bin/activate
	@echo "Virtual environment created, run 'source venv/bin/activate' to activate it."

venv/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

clean:
	rm -rf venv

.PHONY: run clean