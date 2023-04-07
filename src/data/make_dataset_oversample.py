from pathlib import Path

from dotenv import find_dotenv, load_dotenv
import os.path
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import FeatureEngineer
from src.data.feature_engineering_optimised import FeatureEngineerOptimised
from src.data.preprocess import Preprocessor

def bert_aug(Xy_train,TOPK=20, ACT = 'insert'):
    #TOPK: default=100
    #ACT: default="substitute" 
    samples = abs(Xy_train.Sentiment.value_counts()[0] - Xy_train.Sentiment.value_counts()[1]) #count the number of samples to match majority class
    aug_bert = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased', 
        #device='cuda',
        action=ACT, top_k=TOPK)
    aug_bert.aug_p=0.2
    new_text=[]
    ##selecting the minority class samples
    df_n=Xy_train[Xy_train.Sentiment=='negative'].reset_index(drop=True)
    ## data augmentation loop
    for i in tqdm(np.random.randint(0,len(df_n),samples)):
            text = df_n.iloc[i]['Text']
            augmented_text = aug_bert.augment(text)
            new_text.append(augmented_text[0])
    ## dataframe
    new=pd.DataFrame({'Text':new_text,'Sentiment':'negative'})
    new.index = list(range(Xy_train.shape[0]+1,Xy_train.shape[0]+1+samples)) #assign new index values from training set
    new_Xy_train = pd.concat([Xy_train,new]) #combine ori training df and new samples
    return new_Xy_train

def main(input_filepath, train_split_output_filepath=None, test_split_output_filepath=None, oversample=False, generate_oversample=False):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    raw_reviews = pd.read_csv(input_filepath)
    #train-test split
    train, test= train_test_split(raw_reviews, test_size=0.2, random_state=4263, stratify=raw_reviews['Sentiment'])
    num_test_rows = test.shape[0]

    #stack test set below train set
    raw_reviews_train_above_test = pd.concat([train,test], axis=0)
    new_filepath = input_filepath.replace('reviews', 'reviews_train_above_test')
    raw_reviews_train_above_test = raw_reviews_train_above_test.to_csv(new_filepath,index=False)

    #generate oversample with raw reviews
    if oversample and generate_oversample: #need time to generate (~2hrs) 
        print("starting oversampling")
        new_Xy_train = bert_aug(train)
        #combine back new train and old test set to proceed for cleaning & FE
        raw_reviews_oversample = pd.concat([new_Xy_train,test],axis=0)
        new_filepath = input_filepath.replace('reviews', 'reviews_oversample')
        raw_reviews_oversample.to_csv(new_filepath,index=False) #save the new oversampled reviews

    elif oversample: #already generated
        print("new updated script!")
        new_filepath = input_filepath.replace('reviews', 'reviews_oversample')
        #check if file exists
        if os.path.isfile(new_filepath) == False:
            #if never generated before, print error message
            print("oversampled file not generated, please change input param: generate_oversample=True")

    #do preprocessing for new raw file
    preprocessor = Preprocessor(new_filepath)
    preprocessor.clean_csv()
    pre_processed_df = preprocessor.clean_df
    feature_engineer = FeatureEngineerOptimised(pre_processed_df) #FeatureEngineer changed
    feature_engineer.add_features()
    feature_engineered_df = feature_engineer.feature_engineered_df
    #removed index (Unnamed) col
    feature_engineered_df = feature_engineered_df.loc[:, ~feature_engineered_df.columns.str.match('unnamed')]
    # Separate target variable (y) and features (X)
    X = feature_engineered_df.drop(["sentiment", "time"], axis=1)
    y = feature_engineered_df["sentiment"]

    X_train = X[:-num_test_rows] #Remove the bottom subset of the dataset -> this was the test set
    X_test = X[-num_test_rows:] #get bottom subset
    y_train = y[:-num_test_rows]
    y_test = y[-num_test_rows:]

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    if train_split_output_filepath and test_split_output_filepath:
        # Write splits to csv
        train = X_train.join(y_train)
        test = X_test.join(y_test)
        train.to_csv(train_split_output_filepath)
        test.to_csv(test_split_output_filepath)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    # relative dir
    input_file = "../../data/raw/reviews.csv"
    train_output_file = "../../data/processed/train_final_processed_reviews.csv"
    test_output_file = "../../data/processed/test_final_processed_reviews.csv"
    X_train, X_test, y_train, y_test = main(input_file, train_output_file, test_output_file)
