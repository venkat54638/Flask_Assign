import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model = joblib.load('best_final_model.pkl')

# Define the feature name mapping to match model expectations
FEATURE_NAME_MAPPING = {
    'age': 'Age',
    'experience': 'Experience',
    'income': 'Income',
    'zip_code': 'ZIP Code',
    'family': 'Family',
    'ccavg': 'CCAvg',
    'education': 'Education',
    'mortgage': 'Mortgage',
    'securities': 'Securities Account',
    'cd_account': 'CD Account',
    'online': 'Online',
    'creditcard': 'CreditCard'
}

# Set up uploads folder
UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Streamlit app
st.title("Loan Default Prediction")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Single Prediction", "Batch Prediction"])

# Single Prediction Page
if page == "Single Prediction":
    st.header("Single Loan Prediction")
    with st.form(key='single_prediction_form'):
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        experience = st.number_input("Experience (years)", min_value=0, max_value=100, step=1)
        income = st.number_input("Income (in thousands)", min_value=0.0, step=0.1)
        zip_code = st.number_input("ZIP Code", min_value=0, step=1)
        family = st.selectbox("Family Size (1-4)", options=[1, 2, 3, 4])
        ccavg = st.number_input("CCAvg (Avg Credit Card Spend)", min_value=0.0, step=0.01)
        education = st.selectbox("Education Level", options=[(1, "Undergrad"), (2, "Graduate"), (3, "Advanced/Professional")], format_func=lambda x: x[1])[0]
        mortgage = st.number_input("Mortgage (in thousands)", min_value=0.0, step=0.1)
        securities = st.selectbox("Securities Account", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        cd_account = st.selectbox("CD Account", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        online = st.selectbox("Online Banking", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        creditcard = st.selectbox("Credit Card", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        submit_button = st.form_submit_button(label="Predict")

        if submit_button:
            try:
                # Create data dictionary
                data = {
                    'age': float(age),
                    'experience': float(experience),
                    'income': float(income),
                    'zip_code': int(zip_code),
                    'family': int(family),
                    'ccavg': float(ccavg),
                    'education': int(education),
                    'mortgage': float(mortgage),
                    'securities': int(securities),
                    'cd_account': int(cd_account),
                    'online': int(online),
                    'creditcard': int(creditcard)
                }

                # Convert to DataFrame and rename columns
                df = pd.DataFrame([data])
                df.rename(columns=FEATURE_NAME_MAPPING, inplace=True)

                # Verify feature names
                expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(FEATURE_NAME_MAPPING.values())
                if not all(col in expected_features for col in df.columns):
                    missing = [col for col in expected_features if col not in df.columns]
                    extra = [col for col in df.columns if col not in expected_features]
                    st.error(f"Feature mismatch: Missing {missing}, Extra {extra}")
                else:
                    # Predict
                    prediction = model.predict(df)[0]
                    probability = model.predict_proba(df)[0][1]
                    prediction_text = 'Approved ✅' if prediction == 1 else 'Rejected ❌'
                    probability_text = f"{probability:.2%}"
                    st.success(f"Prediction: {prediction_text}")
                    st.info(f"Probability: {probability_text}")
            except Exception as e:
                st.error(f"Error processing form data: {str(e)}")

# Batch Prediction Page
elif page == "Batch Prediction":
    st.header("Batch Loan Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            # Save file
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(filepath, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Read data
            df = pd.read_csv(filepath)

            # Ensure CSV columns match model expectations
            expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(FEATURE_NAME_MAPPING.values())
            
            # Drop unexpected columns (e.g., 'ID')
            extra_columns = [col for col in df.columns if col not in expected_features]
            if extra_columns:
                df = df.drop(columns=extra_columns)
                st.write(f"Dropped extra columns: {extra_columns}")

            # Check for missing features
            missing_features = [col for col in expected_features if col not in df.columns]
            if missing_features:
                st.error(f"Feature mismatch in CSV: Missing {missing_features}, Extra {extra_columns}")
            else:
                # Predict
                predictions = model.predict(df)
                df['Loan Prediction'] = ['Approved ✅' if p == 1 else 'Rejected ❌' for p in predictions]
                
                # Display results as a table
                st.subheader("Prediction Results")
                st.dataframe(df.style.set_properties(**{
                    'background-color': '#888a9e',
                    'border-color': '#ddd',
                    'text-align': 'center'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#39648f'), ('color', 'white')]},
                    {'selector': 'td:nth-child(2)', 'props': [('background-color', '#eacf89'), ('color', '#110404')]}
                ]))
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")