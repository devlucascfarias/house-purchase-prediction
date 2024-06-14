import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report

class HousePurchasePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LogisticRegression()
        self.load_data()
        self.train_model()
    
    def load_data(self):
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
        confusion_mat = sk_confusion_matrix(self.y_test, predictions)
        classification_rep = sk_classification_report(self.y_test, predictions)
        return confusion_mat, classification_rep
    
    def predict(self, salary, income_type, own_property):
        parameters = [[salary, income_type, own_property]]
        prediction = self.model.predict(parameters)
        probability = self.model.predict_proba(parameters)
        return prediction[0], probability[0]

def main():
    st.set_page_config(page_title='House Purchase Prediction', page_icon=':house:', layout='centered', initial_sidebar_state='auto')
    
    st.sidebar.title('House Purchase Prediction')
    st.sidebar.write('This app predicts whether a customer is likely to buy a house or not based on a set of data')
    st.sidebar.write('[GitHub](https://github.com/devlucascfarias)')
    
    st.write('Income type: 1 - Salaried, 2 - Self-employed, 3 - Businessman')
    st.write('Own property: 1 - Yes, 0 - No')

    predictor = HousePurchasePredictor('BaseDados_RegressaoLogistica.xlsx')
    
    salary = st.number_input('Enter the client\'s salary', format='%f')
    income_type = st.number_input('Enter the client\'s income type', format='%f')
    own_property = st.number_input('Enter the client\'s property', format='%f')
    
    if st.button('Predict'):
        prediction, probability = predictor.predict(salary, income_type, own_property)
        probability_percentage = probability[prediction] * 100 
        if prediction == 1:
            st.success('The client is likely to buy the house')
            st.write(f'Probability: {probability_percentage:.2f}%')
        else:
            st.error('The client is unlikely to buy the house')
            st.write(f'Probability: {probability_percentage:.2f}%')

if __name__ == '__main__':
    main()
