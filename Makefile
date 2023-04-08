#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = data-dialogue
PYTHON_INTERPRETER = $(shell ./get_python_interpreter.sh)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


.PHONY: install
install:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	pip install -r requirements.txt -U
	pip install -e . --no-deps

.PHONY: black
black:
	black src --line-length=120
	black tests --line-length=120

.PHONY: test_black
test_black:
	black src --line-length=120 --check
	black tests --line-length=120 --check

.PHONY: flake8
flake8:
	flake8 src --ignore=E203,W503,E501,W605,F401 --count --show-source --statistics
	flake8 tests --ignore=E203,W503,E501,W605 --count --show-source --statistics

.PHONY: test_flake8
test_flake8:
	flake8 src --ignore=E203,W503,E501,W605 --count --show-source --statistics --exit-zero
	flake8 tests --ignore=E203,W503,E501,W605 --count --show-source --statistics --exit-zero

.PHONY: isort
isort:
	isort src --profile black --line-length=120
	isort tests --profile black --line-length=120

.PHONY: test_isort
test_isort:
	isort src -c --profile black --line-length=120
	isort tests -c --profile black --line-length=120

.PHONY: mypy
mypy:
	mypy src --ignore-missing-imports
	mypy tests --ignore-missing-imports

## Format files and typecasting
.PHONY: format
format:
	make black
	make flake8
	make isort
	make mypy

.PHONY: test
test:
	pytest tests --cov-report term-missing --cov=tests/

.PHONY: training_pipeline
training_pipeline:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/src/models/sentiment_analysis/training_pipeline.py

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: run
run:
	streamlit run src/app/main.py


## Set up python interpreter environment
create_environment:
	chmod +x get_python_interpreter.sh
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

