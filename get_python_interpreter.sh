#!/bin/bash

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    # In a conda environment
    echo "$(conda run -n $CONDA_DEFAULT_ENV which python3)"
elif [ -n "$VIRTUAL_ENV" ]; then
    # In a virtual environment
    echo "$VIRTUAL_ENV/bin/python3"
else
    # Neither conda nor virtualenv is active
    echo "$(which python3)"
fi