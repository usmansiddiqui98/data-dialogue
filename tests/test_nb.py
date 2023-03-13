import os

import nbformat
from nbconvert.exporters import NotebookExporter
from nbconvert.preprocessors import ClearOutputPreprocessor

# This test case ensures all the notebooks are cleared of output before committing

def test_notebooks():
    exporter = NotebookExporter(preprocessors=[ClearOutputPreprocessor()])
    for file_name in os.listdir("."):
        if file_name.endswith(".ipynb"):
            with open(file_name) as f:
                nb = nbformat.read(f, as_version=4)
            cleared_nb, _ = exporter.from_notebook_node(nb)
            assert nb == cleared_nb, f"{file_name} has output"
