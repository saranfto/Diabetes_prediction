import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))
Stool = pickle.load(open('scaler.pkl', 'rb'))

#tuning
def t1():
    diagnosis =" The person is not diabetic "
def t2():
    diagnosis =" The person is diabetic "
def t3():
    if ( Glucose < 120) :
        t1()
    elif ( SkinThickness < 20 ) :
        t1()
    elif ( Insulin < 75 ) :
        t1()
    elif ( BMI < 32 ) :
        t1()
    elif ( Age < 35 ) :
        t1()
    else :
        t2()

# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    Stooled_data = Stool.transform(input_data_reshaped)
    # print(Stooled_data)
    prediction = loaded_model.predict(Stooled_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    st.subheader("By Saran B ( 2205102 )", anchor=False)
    st.write(
        "An Artificial Intelligence Enthusiasist and a passionate Machine Learning Engineer. \n \n Enter the values to get diagnosed"
    )
    
    
    
    # getting the input data from the user
    
    
Pregnancies = float(st.text_input('Number of Pregnancies', '0'))
Glucose = float(st.text_input('Glucose Level', '0'))
BloodPressure = float(st.text_input('Blood Pressure value', '0'))
SkinThickness = float(st.text_input('Skin Thickness value', '0'))
Insulin = float(st.text_input('Insulin Level', '0'))
BMI = float(st.text_input('BMI value', '0'))
DiabetesPedigreeFunction = float(st.text_input('Diabetes Pedigree Function value', '0'))
Age = float(st.text_input('Age of the Person', '0'))
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    #t3()    
    st.success(diagnosis)
    
    
    
  
    
if __name__ == '__main__':
    main()
