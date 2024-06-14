import streamlit as st

from _HousePurchasePredictor import HousePurchasePredictor

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
