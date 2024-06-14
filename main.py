import streamlit as st

from _HousePurchasePredictor import HousePurchasePredictor

def main():
    st.set_page_config(page_title='House Purchase Prediction', page_icon=':house:', layout='centered', initial_sidebar_state='auto')
    
    st.sidebar.title('House Purchase Prediction')
    st.sidebar.write('This app predicts whether a customer is likely to buy a house or not based on a set of dataframe')

    st.sidebar.title('Logistic Regression Algorithm')

    st.sidebar.write('Logistic Regression is a machine learning algorithm used for classification problems. In this case, it is used to predict whether a customer will buy a house or not (a binary problem: buy or not buy). The sigmoid function, which is the basis of Logistic Regression, is given by the formula:')
    st.sidebar.latex(r'''
    \sigma(x) = \frac{1}{1 + e^{-x}}
    ''')
    st.sidebar.write('[GitHub](https://github.com/devlucascfarias)')

    predictor = HousePurchasePredictor('BaseDados_RegressaoLogistica.xlsx')

    st.title('Logistic Regression :blue[Example]')


    st.latex(r'''
    \sigma(x) = \frac{1}{1 + e^{-x}}
    ''')

    salary = st.number_input('Enter the client\'s salary', format='%f')

    income_type_options = {
        'Salaried': 1,
        'Self-employed': 2,
        'Businessman': 3
    }
    
    property_options = {
        'Yes': 1,
        'No': 0
    }
    
    income_type_text = st.selectbox('Select the client\'s income type', list(income_type_options.keys()))
    own_property_text = st.selectbox('Does the client own a property?', list(property_options.keys()))
    
    income_type = income_type_options[income_type_text]
    own_property = property_options[own_property_text]
    
    if st.button('Predict'):
        prediction, probability = predictor.predict(salary, income_type, own_property)
        probability_percentage = probability[prediction] * 100 
        if prediction == 1:
            st.success(f'The client is likely to buy the house with a {probability_percentage:.2f}% probability', icon="✅")
        else:
            st.error(f'The client is unlikely to buy the house with a {probability_percentage:.2f}% probability', icon="❌")

if __name__ == '__main__':
    main()
