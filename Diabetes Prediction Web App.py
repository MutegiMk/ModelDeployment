# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:20:44 2022

@author: Mutegi
"""

import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open('C:/Users/Mutegi/Desktop/PROJECTS/PROJECTS-ML/trained_model.sav', 'rb'))

# prediction function

def diabetes_prediction(input_data):
    
    # converting input data to an arrray
    input_data_asarray = np.asarray(input_data)
    #resahping the array
    input_reshaped = input_data_asarray.reshape(1,-1)

    #predicting
    prediction = loaded_model.predict(input_reshaped)

    if prediction[0]==0:
        return prediction, 'patient is not diabetic'
    else:
        return prediction, 'patient is diabetic'
    
    
    
def main():
    # Title
    st.title('Diabetes Prediction Web App')
    
    #Getting the input data from the user
    
    Pregnancies = st.text_input("Number of Pregnancies ever had ?")
    Glucose = st.text_input('Glucose Level ?')
    BloodPressure =  st.text_input('Blood Pressure Value ?')
    SkinThickness =  st.text_input('Skin Thickness value ?')
    Insulin =  st.text_input('Insulin level ?')
    BMI = st.text_input('BMI value ?')
    DiabetesPedigreeFunction =  st.text_input('Diabetes Pedigree Function ?')
    Age = st.text_input('Age of the Person ?')
    
    # Prediction code
    
    diagnosis = ''
    
    # Button creation fro prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure,
                                         SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction,
                                         Age])
        
        
    st.success(diagnosis)
    
        
        
        
if __name__ == '__main__':
    main()
        
    

