Data Dialogue
==============================

Whether for a B2B or B2C company, relevance and longevity in the industry depends on how well the products answer the needs of the customers. However, when the time comes for the companies to demonstrate that understanding — during a sales conversation, customer service interaction, or through the product itself — how can companies evaluate how they measure up?

## To run the app locally end to end:

### Repo:

1. Clone the repo using `git clone <git repo link>` or download the zip file.
2. (Optional) Run `make create_environment` to create a virtual environment.
3. Run `make install` to install dependencies

### Training pipeline:  

1. Upload `bert_state_dict_new_raw.pt` into the models/sentiment_analysis/bert_fine_tuned folder.
2. Run `make train_sentiment` to feature_engineer, preprocess, and train the models

### Scoring pipeline:
1. Run `make run` to predict sentiment based on trained models
2. Open <localhost:8501> in your browser
3. Upload your csv with to predict sentiment

## Scoring API with docker:
This method should be used to skip the training process.
### With GPU:
1. `docker pull tanyx43/data-dialogue:final-gpu2`
2. `docker run --gpus all -it --rm -p 8501:8501 tanyx43/data-dialogue:final-gpu2`
3. Go to <localhost:8501> in your browser
4. Upload your csv with to predict sentiment

### With CPU:
1. `docker pull tanyx43/data-dialogue:final-cpu2`
2. `docker run -p 8501:8501 tanyx43/data-dialogue:final-cpu2`
3. Go to <localhost:8501> in your browser
4. Upload your csv with to predict sentiment

### On an EC2 instance:
1. Install Nvidia Driver and Docker on the EC2 instance
2. Install Nvidia Container Toolkit
3. `docker pull tanyx43/data-dialogue:final-gpu2`
4. `docker run --gpus all -it --rm -p 8501:8501 tanyx43/data-dialogue:final-gpu2`
5. In a new terminal window, enable port forwarding by `ssh -i pem_key.pem ubuntu@<DNS Instance> -L 8501:172.17.0.2:8501`
6. Go to <localhost:8501> in your browser

## Documentation:
<https://usmansiddiqui98.github.io/data-dialogue/>


##  Project Organization
```
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── embeddings
│   │   └── BERTopic_embeddings.pickle
│   ├── predictions
│   │   └── reviews_test_predictions_data-dialogue.csv
│   ├── processed
│   │   ├── clean_reviews_w_topics.csv
│   │   ├── test_final_processed_reviews.csv
│   │   ├── train_final_processed_reviews.csv
│   │   └── train_oversample_final_processed_reviews.csv
│   └── raw
│       ├── reviews.csv
│       ├── reviews_test.csv
│       └── stopwords.txt
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── data
│   │   ├── feature_engineering.rst
│   │   ├── feature_engineering_helpers.rst
│   │   ├── main.rst
│   │   ├── make_dataset.rst
│   │   └── preprocess.rst
│   ├── data.rst
│   ├── getting-started.rst
│   ├── index.rst
│   ├── make.bat
│   ├── models.rst
│   ├── pipelines.rst
│   ├── sentiment_analysis_models
│   │   ├── bert_fine_tuned.rst
│   │   ├── log_reg.rst
│   │   ├── lstm.rst
│   │   ├── main.rst
│   │   ├── naive_bayes.rst
│   │   ├── siebert.rst
│   │   ├── svm.rst
│   │   ├── xg_boost.rst
│   │   └── xg_boost_svd.rst
│   ├── sentiment_analysis_pipelines
│   │   ├── main.rst
│   │   ├── scoring_pipeline.rst
│   │   └── training_pipeline.rst
│   ├── topic_modelling_models
│   │   ├── lda.rst
│   │   ├── lsa.rst
│   │   ├── main.rst
│   │   └── nmf.rst
│   └── topic_modelling_pipelines
│       ├── main.rst
│       └── training_pipeline.rst
├── models
│   ├── sentiment_analysis
│   │   ├── best_model
│   │   │   ├── best_model_name.txt
│   │   │   ├── dim_reduce.pkl
│   │   │   ├── model.pkl
│   │   │   ├── model_class.pkl
│   │   │   └── vectorizer.pkl
│   │   ├── log_reg
│   │   │   ├── model.pkl
│   │   │   ├── scaler.pkl
│   │   │   └── vectorizer.pkl
│   │   ├── lstm
│   │   ├── naive_bayes
│   │   │   ├── model.pkl
│   │   │   └── vectorizer.pkl
│   │   ├── siebert
│   │   ├── svm
│   │   │   ├── model.pkl
│   │   │   └── vectorizer.pkl
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
│   │   ├── huggingface.ipynb
│   │   ├── naive_bayes.ipynb
│   │   └── svm.ipynb
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
│   ├── app
│   │   ├── __init__.py
│   │   ├── _pages
│   │   │   ├── __init__.py
│   │   │   ├── home.py
│   │   │   ├── main.py
│   │   │   ├── sentiment_analysis
│   │   │   │   ├── __init__.py
│   │   │   │   ├── main.py
│   │   │   │   └── sentiment_model.py
│   │   │   └── topic_modelling
│   │   │       ├── __init__.py
│   │   │       ├── evaluation.py
│   │   │       └── main.py
│   │   ├── main.py
│   │   ├── navigation.py
│   │   └── utils.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── generate_oversample.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── sentiment_analysis
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   ├── log_reg.py
│   │   │   ├── lstm.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── pre_trained
│   │   │   │   ├── bert_fine_tuned.py
│   │   │   │   └── siebert.py
│   │   │   ├── scoring_pipeline.py
│   │   │   ├── svm.py
│   │   │   ├── training_pipeline.py
│   │   │   ├── xg_boost.py
│   │   │   └── xg_boost_svd.py
│   │   └── topic_modelling
│   │       ├── LDA.py
│   │       ├── LSA.py
│   │       ├── NMF.py
│   │       ├── __init__.py
│   │       ├── bert_topic.py
│   │       ├── topic_modelling_lda.csv
│   │       └── training_pipeline.py
│   └── utils
│       ├── __init__.py
│       ├── clear_cache.py
│       └── feature_engineering_helpers.py
├── src.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── test_files
│   ├── log_reg_test
│   ├── test_reviews.csv
│   └── train_reviews.csv
├── tests
│   ├── __init__.py
│   ├── test_bert_fine_tuned.py
│   ├── test_environment.py
│   ├── test_feature_engineering.py
│   ├── test_lda.py
│   ├── test_log_reg.py
│   ├── test_lsa.py
│   ├── test_lstm.py
│   ├── test_naive_bayes.py
│   ├── test_nb.py
│   ├── test_nmf.py
│   ├── test_preprocessing.py
│   ├── test_siebert.py
│   ├── test_svm.py
│   ├── test_xg_boost_svd.py
│   └── test_xgboost.py
└── topic_modelling_lda.csv
```
