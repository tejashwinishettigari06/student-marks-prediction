import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("students_data.csv")

# Prepare data
X = data[['hours_study', 'attendance', 'previous_marks']]
y = data['marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Title
st.title("Student Marks Prediction")

# Inputs
hours = st.number_input("Enter study hours")
attendance = st.number_input("Enter attendance")
previous = st.number_input("Enter previous marks")

# Button
if st.button("Predict"):
    input_data = pd.DataFrame([[hours, attendance, previous]],
                              columns=['hours_study', 'attendance', 'previous_marks'])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Marks: {prediction[0]:.2f}")