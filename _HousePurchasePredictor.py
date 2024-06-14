import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 

class HousePurchasePredictor:
    def __init__(self, data_path):
        self.data_path = data_path 
        self.model = LogisticRegression()
        self.model_data()
        self.train_model()

    def model_data(self):
        dataframe = pd.read_excel(self.data_path)
        self.characteristics = dataframe.iloc[:, 1:4].values
        self.predictor = dataframe.iloc[:, 4].values


    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.characteristics, self.predictor, test_size=0.20)
        self.model.fit(x_train, y_train)
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        confusion_mat = confusion_matrix(self.y_test, predictions)
        classification_rep = classification_report(self.y_test, predictions)
        return confusion_mat, classification_rep

    def predict(self, salary, income_type, own_property):
        parameters = [[salary, income_type, own_property]]
        prediction = self.model.predict(parameters)
        probability = self.model.predict_proba(parameters)
        return prediction[0], probability[0]


