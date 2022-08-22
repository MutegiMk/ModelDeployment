# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/Faith/Desktop/PROJECTS/PROJECTS-ML/trained_model.sav', 'rb'))

# making a predictive model
input_data = (5,166,80,19,190,26.8,0.486,69)
# converting input data to an arrray
input_data_asarray = np.asarray(input_data)
#resahping the array
input_reshaped = input_data_asarray.reshape(1,-1)

#predicting
prediction = loaded_model.predict(input_reshaped)

if prediction[0]==0:
    print(prediction, 'patient is not diabetic')
else:
    print(prediction, 'patient is diabetic')