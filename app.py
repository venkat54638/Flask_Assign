from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Load model
model = joblib.load('best_final_model.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

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

# Route for the home page (form for single prediction)
@app.route('/')
def home():
    return render_template('index.html')

# Route for single prediction from form inputs
@app.route('/predict', methods=['POST'])
def predict():
    if 'age' in request.form:
        try:
            # Extract form data
            data = {
                'age': float(request.form['age']),
                'experience': float(request.form['experience']),
                'income': float(request.form['income']),
                'zip_code': int(request.form['zip_code']),
                'family': int(request.form['family']),
                'ccavg': float(request.form['ccavg']),
                'education': int(request.form['education']),
                'mortgage': float(request.form['mortgage']),
                'securities': int(request.form['securities']),
                'cd_account': int(request.form['cd_account']),
                'online': int(request.form['online']),
                'creditcard': int(request.form['creditcard'])
            }

            # Convert to DataFrame and rename columns to match model expectations
            df = pd.DataFrame([data])
            df.rename(columns=FEATURE_NAME_MAPPING, inplace=True)

            # Verify feature names match model expectations
            expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(FEATURE_NAME_MAPPING.values())
            if not all(col in expected_features for col in df.columns):
                missing = [col for col in expected_features if col not in df.columns]
                extra = [col for col in df.columns if col not in expected_features]
                return f"Feature mismatch: Missing {missing}, Extra {extra}", 400

            # Predict
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]  # Probability of positive class

            # Format prediction
            prediction_text = 'Approved ✅' if prediction == 1 else 'Rejected ❌'
            probability_text = f"{probability:.2%}"

            return render_template('index.html', prediction=prediction_text, probability=probability_text)
        except Exception as e:
            return f"Error processing form data: {str(e)}", 400

    return "Invalid request.", 400

# Route for batch prediction (CSV upload)
@app.route('/predict_batch', methods=['GET', 'POST'])
def predict_batch():
    if request.method == 'GET':
        return render_template('batch.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request.", 400

        file = request.files['file']
        
        if file.filename == '':
            return "No file selected.", 400

        if file:
            try:
                # Save file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Read data
                df = pd.read_csv(filepath)

                # Ensure CSV columns match model expectations
                expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(FEATURE_NAME_MAPPING.values())
                
                # Drop unexpected columns (e.g., 'ID')
                extra_columns = [col for col in df.columns if col not in expected_features]
                if extra_columns:
                    df = df.drop(columns=extra_columns)
                    print(f"Dropped extra columns: {extra_columns}")

                # Check for missing features
                missing_features = [col for col in expected_features if col not in df.columns]
                if missing_features:
                    return f"Feature mismatch in CSV: Missing {missing_features}, Extra {extra_columns}", 400

                # Predict
                predictions = model.predict(df)
                df['Loan Prediction'] = ['Approved ✅' if p == 1 else 'Rejected ❌' for p in predictions]

                # Render predictions as HTML table
                return render_template('predict.html', Y=df.to_html(classes='table table-bordered'))
            except Exception as e:
                return f"Error processing file: {str(e)}", 400

        return "Invalid file.", 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)