import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# Set page configuration
st.set_page_config(layout='wide')

# --- Constants (ensure these match your training setup) ---
HF_MODEL_REPO = 'nareshpaib/tourism_prediction_decision_tree_model'
HF_MODEL_FILENAME = 'best_decision_tree_model.joblib'

# Caching model loading for performance
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME, repo_type='model')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Preprocessing details (MUST match the training preprocessing exactly) ---
categorical_cols_encoded = [
    'TypeofContact_Company Invited',
    'Occupation_Salaried',
    'Occupation_Small Business',
    'Occupation_Student',
    'Gender_Female',
    'Gender_Male',
    'ProductPitched_Deluxe',
    'ProductPitched_Executive',
    'ProductPitched_King',
    'ProductPitched_Standard',
    'MaritalStatus_Married',
    'MaritalStatus_Single'
]

# List of all feature columns the model was trained on (excluding ProdTaken)
# This list should be derived from X_train.columns from the notebook.
# Numerical columns that were not one-hot encoded
original_numerical_cols = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

# Combine all expected features for reindexing
expected_features = original_numerical_cols + categorical_cols_encoded


# --- Streamlit UI --- 
st.title('Wellness Tourism Package Purchase Prediction')
st.write('Enter customer details to predict if they will purchase the Wellness Tourism Package.')

if model is None:
    st.stop()

# Input fields for user
with st.form('prediction_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        contact_type = st.selectbox('Type of Contact', ['Company Invited', 'Self Inquiry'])
        city_tier = st.selectbox('City Tier', [1, 2, 3])
        occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Student'])
        gender = st.selectbox('Gender', ['Male', 'Female', 'Fe Male'])
        num_persons = st.number_input('Number of Persons Visiting', min_value=1, max_value=10, value=2)
        num_followups = st.number_input('Number of Follow-ups', min_value=0, max_value=10, value=3)

    with col2:
        product_pitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Executive', 'King', 'Standard'])
        pref_property_star = st.selectbox('Preferred Property Star', [3, 4, 5])
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
        num_trips = st.number_input('Number of Trips Annually', min_value=0, max_value=50, value=5)
        passport = st.selectbox('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
        own_car = st.selectbox('Owns a Car?', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

    with col3:
        num_children = st.number_input('Number of Children Visiting (below 5)', min_value=0, max_value=5, value=0)
        monthly_income = st.number_input('Monthly Income', min_value=0.0, value=25000.0, step=1000.0)
        duration_pitch = st.number_input('Duration of Pitch (minutes)', min_value=1.0, max_value=60.0, value=10.0)
        pitch_satisfaction_score = st.slider('Pitch Satisfaction Score', min_value=1, max_value=5, value=3)

    submitted = st.form_submit_button('Predict Purchase')

    if submitted:
        # Create a dictionary from inputs
        input_data = {
            'Age': age,
            'TypeofContact': contact_type,
            'CityTier': city_tier,
            'DurationOfPitch': duration_pitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': num_persons,
            'NumberOfFollowups': num_followups,
            'ProductPitched': product_pitched,
            'PreferredPropertyStar': pref_property_star,
            'MaritalStatus': marital_status,
            'NumberOfTrips': num_trips,
            'Passport': passport,
            'PitchSatisfactionScore': pitch_satisfaction_score,
            'OwnCar': own_car,
            'NumberOfChildrenVisiting': num_children,
            'MonthlyIncome': monthly_income
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply one-hot encoding, matching training set columns
        # Recreate the exact categorical columns used during training
        categorical_cols_app = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols_app, drop_first=True)

        # Ensure all columns from training are present and in the correct order
        # Add missing columns (from one-hot encoding) with value 0
        missing_cols = set(expected_features) - set(input_encoded.columns)
        for c in missing_cols:
            input_encoded[c] = 0
        
        # Reindex to ensure the order of columns is the same as training data
        input_processed = input_encoded[expected_features]

        # Make prediction
        prediction = model.predict(input_processed)
        prediction_proba = model.predict_proba(input_processed)[:, 1]

        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.success(f'The model predicts that the customer **WILL** purchase the package with a probability of {prediction_proba[0]:.2f}.')
        else:
            st.info(f'The model predicts that the customer **WILL NOT** purchase the package with a probability of {prediction_proba[0]:.2f}.')
        
        st.write('---')
        st.write('Input Data (Processed for Model):')
        st.dataframe(input_processed)
