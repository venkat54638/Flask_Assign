###  Bank Personal Loan Modelling

---

## Project Overview

This project aims to predict whether a customer will accept a bank loan based on historical data. Using machine learning techniques, we preprocess data, train predictive models, and deploy them via an API endpoint for real-time predictions.

---

###  Features

1. **Data Ingestion**: Loads data from Excel files into DataFrame objects.
2. **Preprocessing**: Handles missing values, encodes categorical variables, scales numerical features.
3. **Model Selection**: Uses decision trees and grid search for hyperparameter tuning.
4. **Training & Deployment**: Trains a model and deploys it using Streamlit and Flask APIs.
5. **Predictive Endpoints**: Exposes an API endpoint for making predictions using the trained model.

---

###  Prerequisites

To work with this project, you'll need the following tools and libraries installed:

- Python (>=3.8)
- Pandas, NumPy
- Scikit-learn
- Flask
- Joblib
- OpenSSL (for serving HTTPS)

You also need the `Bank_Personal_Loan_Modelling.xlsx` and `CRISP_ML(Q).xlsx` datasets located in your working directory.

---

###  Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo.git
   ```

2. Install the required dependencies:

   ```bash
   pip install pandas numpy jupyterlab scikit-learn flask joblib openpyxl
   ```

3. Copy the sample dataset into your local directory (`Bank_Personal_Loan_Modelling.xlsx`).

4. Start the development server:

   ```bash
   python app.py
   ```

   This will serve the application locally.

---

###  Usage Examples

#### **Example 1: Training the Model**

After preprocessing the data, follow these steps to train the model:

1. Import necessary modules:

   ```python
   import data_ingest
   import preprocessing
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   ```

2. Load and preprocess the data:

   ```python
   df = data_ingest.data_load('Bank_Personal_Loan_Modelling.xlsx', 'Sheet1')
   
   df = preprocessing.preprocess(df)
   ```

3. Split the data and set targets:

   ```python
   X = df.drop(columns=['Target']).features  # Replace 'Target' with your target variable
   y = df['Target']
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
   ```

4. Train the model:

   ```python
   clf = DecisionTreeClassifier(random_state=...)
   clf.fit(X_train, y_train)
   ```

5. Evaluate and save the model:

   ```python
   joblib.dump(clf, 'loan_model.pkl')
   ```

#### **Example 2: Making Predictions via API**

Use the deployed API endpoint to submit predictions:

```bash
curl -X POST \
  http://localhost:5000/predict \
  -H "content-type: application/json" \
  --data '{"id": 123, "sex": "male", ...}'} 
```

Replace `{...}` with the JSON-formatted input fields.

---

###  API Documentation

The API exposes two endpoints:

1. **Training Endpoint** : `/train`  
   Deployed as part of the Flask app. Runs the training pipeline upon request.

2. **Prediction Endpoint** : `/predict`  
   Submits predictions based on the saved model.

Endpoints may vary depending on implementation specifics.

---

###  Contributing

Contributions are welcome! To get started:

1. Fork the repository.
2. Implement new features or fix bugs.
3. Write comprehensive test cases.
4. Submit a pull request with appropriate metadata.

---


### Conclusion

This solution provides a robust framework for loan prediction, combining data processing, machine learning, and API-based deployment. It’s designed to handle dynamic changes efficiently while maintaining scalability and performance.
