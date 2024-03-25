import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_cleaned.csv")

# Preprocess the data
def preprocess_data(df):
    # Handle categorical columns
    label_encoder = LabelEncoder()
    df['sex'] = label_encoder.fit_transform(df['sex'])
    df['fbs'] = label_encoder.fit_transform(df['fbs'])
    df['exang'] = label_encoder.fit_transform(df['exang'])
    
    # Check if 'target' column exists before dropping
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    
    return df

# Train the model
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Main function to run the app
def main():
    # Page title
    st.title("Heart Disease Prediction")

    # Load data
    data = load_data()

    # Preprocess data
    data_processed = preprocess_data(data)

    # Train model
    X = data_processed.drop(columns=['target']) if 'target' in data_processed.columns else data_processed
    y = data['target'] if 'target' in data.columns else None
    model = train_model(X, y)

    # Sidebar with user inputs
    st.sidebar.title("Enter Patient Information")
    age = st.sidebar.slider("Age", min_value=0, max_value=100, step=1)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, step=1)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, step=1)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=0, max_value=300, step=1)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, step=1)
    thal = st.sidebar.selectbox("Thalassemia Type", [0, 1, 2, 3])

    # Make prediction
    patient_info = [[age, 1 if sex == "Male" else 0, cp, trestbps, chol, 1 if fbs == "Yes" else 0, restecg, thalach, 1 if exang == "Yes" else 0, oldpeak, slope, ca, thal]]
    prediction = model.predict(patient_info)

    # Display prediction
    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.write("The patient is predicted to not have heart disease.")
    else:
        st.write("The patient is predicted to have heart disease.")

if __name__ == "__main__":
    main()
