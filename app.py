import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("titanic_model.pkl", "rb"))

# App title
st.title("Titanic Survival Prediction App")
st.write("Predict if a passenger survived the Titanic disaster.")

# Load dataset for EDA
train = pd.read_csv("train.csv")

# EDA Section
st.header("EDA Highlights")
st.write("Survival Counts:")
st.bar_chart(train['Survived'].value_counts())
st.write("Survival by Sex:")
st.bar_chart(train.groupby('Sex')['Survived'].mean())
st.write("Survival by Passenger Class:")
st.bar_chart(train.groupby('Pclass')['Survived'].mean())

# Prediction inputs
st.header("Predict Survival")
age = st.number_input("Age", 0, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class", [1,2,3])
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
has_cabin = st.selectbox("Has Cabin?", [0,1])
embarked_Q = st.selectbox("Embarked_Q", [0,1])
embarked_S = st.selectbox("Embarked_S", [0,1])

# Convert sex to numeric
sex_num = 0 if sex == "male" else 1

# Create dataframe for prediction (columns must match training)
user_data = pd.DataFrame({
    'Pclass':[pclass],
    'Sex':[sex_num],
    'Age':[age],
    'SibSp':[sibsp],
    'Parch':[parch],
    'Fare':[fare],
    'HasCabin':[has_cabin],
    'Embarked_Q':[embarked_Q],
    'Embarked_S':[embarked_S]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(user_data)
    result = "Survived!" if prediction[0] == 1 else "Did not survive."
    st.success(result)

# Show model accuracy
st.header("Model Accuracy")
st.write("Accuracy on Test Set: 81%")
