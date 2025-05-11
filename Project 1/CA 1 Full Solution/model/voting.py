import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from Config import *
from utils import parse_full_type
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
eclf1 = VotingClassifier(estimators=[
         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Voting(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(Voting, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = eclf1
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        ytest = pd.Series(data.y_test).astype("string")
        ypred = pd.Series(self.predictions).astype("string")
        test_df = pd.concat([ytest, ypred], axis=1)
        test_df.columns = ['full_type', 'pred_full_type']
        test_df[Config.PRED_TYPE_COLS] = test_df['pred_full_type'].apply(parse_full_type)
        test_df[Config.FORMATTED_TYPE_COLS] = test_df['full_type'].apply(parse_full_type)
        from sklearn.metrics import accuracy_score
        accuracies = [accuracy_score(test_df[true_col], test_df[pred_col])
                      for true_col, pred_col in zip(Config.FORMATTED_TYPE_COLS, Config.PRED_TYPE_COLS)]
        print(accuracies)


    def data_transform(self) -> None:
        ...

