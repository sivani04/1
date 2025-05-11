import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from Config import *
from utils import parse_full_type

class Hist_GB(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(Hist_GB, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = HistGradientBoostingClassifier()
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

