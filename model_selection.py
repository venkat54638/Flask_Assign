import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os
 
def evaluate_models_with_grid_search(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
            "params": {
                "C": [0.01, 0.1, 1, 10]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
            "params": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        }
    }
 
    best_score = 0
    best_overall_model = None
    best_overall_params = None
 
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Loan_Default_Classification")
 
    for name, item in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\n🔍 Training {name} with GridSearchCV...")
 
            grid = GridSearchCV(item["model"], item["params"], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
 
            best_model = grid.best_estimator_
            preds = best_model.predict(X_test)
 
            # Evaluation metrics
            acc = accuracy_score(y_test, preds)
            prec = classification_report(y_test, preds, output_dict=True)['1']['precision']
            rec = classification_report(y_test, preds, output_dict=True)['1']['recall']
            f1 = classification_report(y_test, preds, output_dict=True)['1']['f1-score']
            # Log to MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })
            mlflow.sklearn.log_model(best_model, artifact_path="model")
 
            mlflow.set_tag("model_name", name)

            model_path = "model"
            mlflow.sklearn.log_model(best_model, artifact_path=model_path)
            model_uri = f"runs:/{run.info.run_id}/{model_path}"
            mlflow.set_tag("model_name", name)
 
            # Console output
            print(f"\n Best Params for {name}: {grid.best_params_}")
            print(f" Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            print(f" Classification Report:\n{classification_report(y_test, preds)}")
            print("=" * 60)
 
            if acc > best_score:
                best_overall_model = best_model
                best_overall_params = grid.best_params_
                best_score = acc
                best_model_uri = model_uri
                best_run_id = run.info.run_id
        if best_model_uri:
            model_name = "Loan_Default_Best_Model"
            print(f"\nRegistering the best model to MLflow Model Registry as '{model_name}'...")
            mlflow.register_model(model_uri=best_model_uri, name=model_name)
            print("Model registered successfully.")
    return best_overall_model, best_overall_params