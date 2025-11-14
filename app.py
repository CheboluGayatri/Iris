import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("IRIS.csv")

X = df.drop("species", axis=1)
y = df["species"]

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("Iris Flower Classifier")
st.write("Enter the measurements to predict the species.")

sepal_length = st.number_input("Sepal length", min_value=0.0)
sepal_width = st.number_input("Sepal width", min_value=0.0)
petal_length = st.number_input("Petal length", min_value=0.0)
petal_width = st.number_input("Petal width", min_value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.success(f"Predicted species: {prediction}")
