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
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_sentiment_features.py
    │   │   └── build_topic_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── sentiment_analysis
    │   │   │   ├── pre_trained
    │   │   │   │   └── predict_bert.py
    │   │   └── topic_modelling
    │   │       ├── lda
    │   │       │   └── predict_lda.py
    │   │       └── lsa
    │   │           └── predict_lsa.py
    │── tests        <- All tests for the project
    │   ├── __init__.py    <- Makes tests a Python module
    │   ├── test_sentiment_analysis.py
    │   └── test_topic_modelling.py
    │── Dockerfile         <- Dockerfile for building the app
    │── Makefile          <- Makefile with commands like `make install` or `make run`
    │── pre-commit-config.yaml <- pre-commit configuration file
    │── .flake8            <- flake8 configuration file
    │── requirements.txt   <- The requirements file for reproducing the analysis environment
    │── setup.py           <- makes project pip installable (pip install -e .) so src can be imported

--------
