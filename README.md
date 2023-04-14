Data Dialogue
==============================

Whether for a B2B or B2C company, relevance and longevity in the industry depends on how well the products answer the needs of the customers. However, when the time comes for the companies to demonstrate that understanding — during a sales conversation, customer service interaction, or through the product itself — how can companies evaluate how they measure up?

## To run the app locally:

1. Clone the repo using `git clone` or download the zip file.
2. Run `make create_environment` to create a virtual environment.
3. Run `make install` to install dependencies
4. Upload `bert_state_dict_new_raw.pt` into the models/sentiment_analysis/bert_fine_tuned folder.
5. Run `make training_pipeline` to feature_engineer, preprocess, and train the models
6. Run `make run` to start the app to predict the sentiment.
7. Go to `localhost:8501` in your browser
8. Upload a file to predict the sentiment


## To change the model run by our Scoring Pipeline:
1. Go to `src/app/_pages/sentiment_analysis/sentiment_model.py`
2. Change the `best_model` variable to the name of the model you want to use. The model names are the same as the names of the files in `src/models/sentiment_analysis/`

##  Project Organization
```
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── processed
│   │   ├── clean_reviews_w_topics.csv
│   │   ├── test_final_processed_reviews.csv
│   │   └── train_final_processed_reviews.csv
│   └── raw
│       ├── reviews.csv
│       ├── reviews_test.csv
│       └── stopwords.txt
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── models
│   ├── sentiment_analysis
│   │   ├── best_model
│   │   │   ├── dim_reduce.pkl
│   │   │   ├── model.pkl
│   │   │   ├── model_class.pkl
│   │   │   └── vectorizer.pkl
│   │   ├── log_reg
│   │   │   ├── model.pkl
│   │   │   └── vectorizer.pkl
│   │   ├── seibert
│   │   ├── xg_boost
│   │   │   ├── model.pkl
│   │   │   └── vectorizer.pkl
│   │   └── xg_boost_svd
│   │       ├── dim_reduce.pkl
│   │       ├── model.pkl
│   │       └── vectorizer.pkl
│   └── topic_modelling
├── notebooks
│   ├── eda
│   │   ├── data_common_grams.csv
│   │   └── eda_raw_data.ipynb
│   ├── pre_processing.ipynb
│   ├── presentation
│   │   └── Sentiment_Analysis.ipynb
│   ├── sentiment
│   │   ├── BERT.ipynb
│   │   ├── BasicLSTM.ipynb
│   │   ├── LSTM_with_GloVe.ipynb
│   │   ├── XGBoost.ipynb
│   │   ├── XGboost_2.ipynb
│   │   ├── baselines.ipynb
│   │   ├── feature_engineering.ipynb
│   │   └── huggingface.ipynb
│   └── topic modelling
│       ├── BERT.ipynb
│       ├── BERTopic.ipynb
│       ├── BERTopic_embeddings.pickle
│       ├── LDA_LSA.ipynb
│       ├── NMF.ipynb
│       ├── reviews_all-MiniLM-L6-v2_embedding.pickle
│       └── reviews_all-mpnet-base-v2_embedding.pickle
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-310.pyc
│   ├── app
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── navigation.cpython-310.pyc
│   │   │   └── utils.cpython-310.pyc
│   │   ├── _pages
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   ├── home.cpython-310.pyc
│   │   │   │   └── main.cpython-310.pyc
│   │   │   ├── home.py
│   │   │   ├── main.py
│   │   │   ├── sentiment_analysis
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   │   ├── main.cpython-310.pyc
│   │   │   │   │   └── sentiment_model.cpython-310.pyc
│   │   │   │   ├── main.py
│   │   │   │   └── sentiment_model.py
│   │   │   └── topic_modelling
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-310.pyc
│   │   │       │   ├── bert_topic.cpython-310.pyc
│   │   │       │   └── main.cpython-310.pyc
│   │   │       ├── bert_topic.py
│   │   │       └── main.py
│   │   ├── main.py
│   │   ├── navigation.py
│   │   └── utils.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── feature_engineering.cpython-310.pyc
│   │   │   ├── feature_engineering_optimised.cpython-310.pyc
│   │   │   ├── make_dataset.cpython-310.pyc
│   │   │   └── preprocess.cpython-310.pyc
│   │   ├── feature_engineering.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-310.pyc
│   │   ├── sentiment_analysis
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   ├── base_model.cpython-310.pyc
│   │   │   │   ├── log_reg.cpython-310.pyc
│   │   │   │   ├── training_pipeline.cpython-310.pyc
│   │   │   │   ├── xg_boost.cpython-310.pyc
│   │   │   │   └── xg_boost_svd.cpython-310.pyc
│   │   │   ├── base_model.py
│   │   │   ├── bert.py
│   │   │   ├── log_reg.py
│   │   │   ├── pre_trained
│   │   │   │   ├── __pycache__
│   │   │   │   │   └── seibert.cpython-310.pyc
│   │   │   │   ├── bert_fine_tuned.py
│   │   │   │   └── seibert.py
│   │   │   ├── scoring_pipeline.py
│   │   │   ├── training_pipeline.py
│   │   │   ├── xg_boost.py
│   │   │   └── xg_boost_svd.py
│   │   └── topic_modelling
│   │       ├── LDA.py
│   │       ├── LSA.py
│   │       ├── NMF.py
│   │       └── __init__.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   └── feature_engineering_helpers.cpython-310.pyc
│       ├── clear_cache.py
│       └── feature_engineering_helpers.py
├── src.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── test_files
│   └── log_reg_test
└── tests
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   ├── test_dummy.cpython-310-pytest-7.2.1.pyc
    │   ├── test_dummy.cpython-310-pytest-7.2.2.pyc
    │   ├── test_environment.cpython-310-pytest-7.2.2.pyc
    │   ├── test_feature_engineering.cpython-310-pytest-7.2.2.pyc
    │   ├── test_log_reg.cpython-310-pytest-7.2.2.pyc
    │   ├── test_nb.cpython-310-pytest-7.2.2.pyc
    │   ├── test_preprocess.cpython-310-pytest-7.2.2.pyc
    │   ├── test_preprocess_feat_engineering.cpython-310-pytest-7.2.2.pyc
    │   └── test_preprocessing.cpython-310-pytest-7.2.2.pyc
    ├── test_environment.py
    ├── test_feature_engineering.py
    └── test_nb.py
```
