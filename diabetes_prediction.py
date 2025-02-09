import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

import streamlit as st
from PIL import Image


def predict(input_data: list):
    try:
        # Load the model and scaler
        model = pickle.load(open('diabetes_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the saved scaler

        # Convert input data to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  # Reshape for single prediction

        # Standardize the input data using the pre-fitted scaler
        std_data = scaler.transform(input_data_reshaped)

        # Make prediction
        prediction = model.predict(std_data)

        # Print the result
        return prediction[0]

    except Exception as e:
        return f"An error occurred: {e}"

# Test the function
# predict([6, 148, 72, 35, 0, 33.6, 0.627, 50])  # The person is diabetic
# predict([1, 85, 66, 29, 0, 26.6, 0.351, 31])  # The person is not diabetic

def main():
    # Custom CSS for styling the title and "Created by" section
    st.markdown("""
    <style>
    .title {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #4CAF50 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    .created-by {
        font-size: 18px !important;
        font-style: italic;
        color: #555555;
        text-align: center;
    }
    .profile-pic {
        border-radius: 50%;
        width: 100px;
        height: 100px;
        object-fit: cover;
        margin: 10px auto;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title with custom design
    st.markdown('<p class="title">Diabetes Prediction App</p>', unsafe_allow_html=True)

    # Create a sidebar for "Created by" and profile picture
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display your profile picture
        profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
        st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG", width=100)

    # App description
    st.write("""
    This app predicts whether a person has diabetes based on their health metrics.
    Please fill in the details below and click **Predict**.
    """)

    # Create a two-column layout for input fields
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 17, 3, help="Number of times pregnant")
        glucose = st.number_input("Glucose", 0, 199, 117, help="Plasma glucose concentration (mg/dL)")
        blood_pressure = st.number_input("Blood Pressure", 0, 122, 72, help="Diastolic blood pressure (mm Hg)")
        skin_thickness = st.number_input("Skin Thickness", 0, 99, 23, help="Triceps skinfold thickness (mm)")

    with col2:
        insulin = st.number_input("Insulin", 0, 846, 30, help="2-Hour serum insulin (mu U/ml)")
        bmi = st.number_input("BMI", 0.0, 67.1, 32.0, help="Body mass index (weight in kg/(height in m)^2)")
        dpf = st.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.3725, help="Diabetes pedigree function")
        age = st.number_input("Age", 21, 81, 29, help="Age in years")

    # Make a list of the input data
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    # Predict the result when the button is clicked
    if st.button("Predict"):
        with st.spinner("Predicting..."):  # Show a loading spinner
            result = predict(input_data)

        # Display the result with styling
        if result == 0:
            st.success("The person is **not diabetic**. 🎉")
        else:
            st.error("The person is **diabetic**. 🚨")

    st.sidebar.title("About")
    st.sidebar.info("This app uses a machine learning model to predict diabetes risk.")
    st.sidebar.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

if __name__ == "__main__":
    main()
    