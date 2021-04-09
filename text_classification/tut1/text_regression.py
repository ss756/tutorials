import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model
import os
import sys
import glob
from sklearn.feature_extraction.text import CountVectorizer


def Train(df):
    # map all the positive sentiments to 1
    # map all the negative sentiments to 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    # we create a new column in the dataset and fill it with -1
    df["kfolds"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.sentiment.values
    kf =model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        print(f, t_, v_)
        df.loc[v_,'kfold'] = f

    # we go over the folds created
    for fold_ in range(5):
        print(f"Fold: {fold_}")
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

    #Initialize CountVectorizer with NLTK's word_tokenize
    #function as tokenizer

        count_vec = CountVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
        )

        # fit count_vec on training data reviews
        count_vec.fit(train_df.review)

        #transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        #initialize the logistic regression model
        model = linear_model.LogisticRegression()
        model.fit(xtrain, train_df.sentiment)

        # make predictions on the test data
        preds = model.predict(
            xtest
        )
        # calculate the accuracy of the model
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold : {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")














if __name__ == "__main__":
    df = pd.read_csv("./data/IMDB Dataset.csv")
    Train(df)

