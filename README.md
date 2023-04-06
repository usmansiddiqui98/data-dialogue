Data Dialogue
==============================

Whether for a B2B or B2C company, relevance and longevity in the industry depends on how well the products answer the needs of the customers. However, when the time comes for the companies to demonstrate that understanding — during a sales conversation, customer service interaction, or through the product itself — how can companies evaluate how they measure up?

## To run the app locally:
1. Clone the repo using `git clone` or download the zip file.
2. Run `make install` to install dependencies
3. Run `make run` to start the app
4. Go to `localhost:8501` in your browser

Project Organization
------------

    ├── LICENSE
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to preprocess, feature engineer, and split data
    │   │   ├── feature_engineering.py
    │   │   ├── make_dataset.py
    │   │   ├── preprocess.py
    │   │
    │   ├── models         <- Scripts to train models and predict using trained models
    │   │   ├── sentiment_analysis
    │   │   │   ├── base_model.py
    │   │   └── topic_modelling
    │   │
    │   ├── utils         <- Scripts to perform common tasks
    │   │   ├── clear_cache.py
    │   │   ├── feature_engineering.py
    │
    │── tests        <- All tests for the project
    │   ├── __init__.py    <- Makes tests a Python module
    │   ├── test_environment.py
    │   ├── test_feature_engineering.py
    │   ├── test_nb.py
    │   ├── test_preprocess.py
    │   ├── test_preprocess_feat_engineering.py
    │
    │── Dockerfile         <- Dockerfile for building the app
    │
    │── Makefile          <- Makefile with commands like `make install` or `make run`
    │
    │── pre-commit-config.yaml <- pre-commit configuration file
    │    
    │── .flake8            <- flake8 configuration file
    │    
    │── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    │── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
--------
