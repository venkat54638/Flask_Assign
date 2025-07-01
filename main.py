import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import data_ingest
import preprocessing
import model_selection
import pickle
import os
import train_save_model

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load and preprocess data
df = data_ingest.data_load('Bank_Personal_Loan_Modelling.xlsx', 'Data')
df = preprocessing.preprocess(df)

# Split data
X = df.drop(['Personal Loan', 'ID'], axis=1)
y = df['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run model evaluation and find best model
best_model, best_params = model_selection.evaluate_models_with_grid_search(X_train, X_test, y_train, y_test)

# Train and save final model
def train_and_save_final_model(X, y, model, model_name="best_final_model.pkl"):
    with mlflow.start_run(run_name="Final Model Training"):
        # Train model on full data
        model.fit(X, y)
        
        # Evaluate on test set (optional, for logging metrics)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Log model name as a tag
        mlflow.set_tag("model_name", model.__class__.__name__)
        
        # Save model locally (optional, for backward compatibility)
        with open(model_name, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Model trained and logged to MLflow with run ID: {mlflow.active_run().info.run_id}")
        print(f"   Model saved locally as: {model_name}")
        return model

# Train and save the final model
train_save_model.train_and_save_final_model = train_and_save_final_model  # Override the function
train_save_model.train_and_save_final_model(X, y, best_model, model_name="best_final_model.pkl")