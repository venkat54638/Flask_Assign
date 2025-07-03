import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from sklearn.model_selection import train_test_split
import mlflow
import os

def log_evidently_metrics(report_dict, prefix=""):
    for metric in report_dict['metrics']:
        metric_name = metric.get('metric')
        result = metric.get('result', {})
        if isinstance(result, dict):
            if 'drift_by_columns' in result:
                for col, val in result['drift_by_columns'].items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(f"{prefix}drift_{col}", val)
            for k, v in result.items():
                if isinstance(v, (int, float)) and k != "drift_by_columns":
                    mlflow.log_metric(f"{prefix}{metric_name}_{k}", v)

def run_evidently_drift_reports():
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(base_dir, "../uploads")
    xls_path = os.path.join(uploads_dir, "Bank_Personal_Loan_Modelling.xlsx")
    new_csv_path = os.path.join(uploads_dir, "New Customer Bank_Personal_Loan.csv")

    # Load and prepare data
    df = pd.read_excel(xls_path, sheet_name='Data')
    X = df.drop(['Personal Loan', 'ID'], axis=1)
    y = df['Personal Loan']
    train_df, test_df, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_df['Personal Loan'] = y_train
    test_df['Personal Loan'] = y_test

    new_df = pd.read_csv(new_csv_path)
    if 'Personal Loan' not in new_df.columns:
        new_df['Personal Loan'] = -1  # Dummy value

    # Define column mappings
    column_mapping_with_target = ColumnMapping(
        target='Personal Loan',
        prediction=None,
        numerical_features=X.select_dtypes(include='number').columns.tolist(),
        categorical_features=X.select_dtypes(include=['object', 'category']).columns.tolist()
    )
    column_mapping_without_target = ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=X.select_dtypes(include='number').columns.tolist(),
        categorical_features=X.select_dtypes(include=['object', 'category']).columns.tolist()
    )

    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LoanDriftMonitoring")

    # 🚀 Run 1: Train vs Test
    with mlflow.start_run(run_name="train vs test"):
        report_test = Report(metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            TargetDriftPreset()
        ])
        report_test.run(reference_data=train_df, current_data=test_df, column_mapping=column_mapping_with_target)
        report_path_1 = os.path.join(base_dir, "report_train_vs_test.html")
        report_test.save_html(report_path_1)
        mlflow.log_artifact(report_path_1, artifact_path="evidently_reports")
        
        test_metrics = report_test.as_dict()
        log_evidently_metrics(test_metrics)

        mlflow.log_metric("rows_train", len(train_df))
        mlflow.log_metric("rows_test", len(test_df))
        mlflow.log_param("dataset", "Bank_Personal_Loan")
        mlflow.log_param("drift_report", "Train vs Test")

    # 🚀 Run 2: Old vs New
    with mlflow.start_run(run_name="old vs new"):
        report_new = Report(metrics=[
            DataQualityPreset(),
            DataDriftPreset()
        ])
        report_new.run(reference_data=train_df, current_data=new_df, column_mapping=column_mapping_without_target)
        report_path_2 = os.path.join(base_dir, "report_old_vs_new.html")
        report_new.save_html(report_path_2)
        mlflow.log_artifact(report_path_2, artifact_path="evidently_reports")

        new_metrics = report_new.as_dict()
        log_evidently_metrics(new_metrics)

        mlflow.log_metric("rows_train", len(train_df))
        mlflow.log_metric("rows_new", len(new_df))
        mlflow.log_param("dataset", "Bank_Personal_Loan")
        mlflow.log_param("drift_report", "Old vs New")

    print("✅ Both drift reports complete and logged to MLflow.")

# Run
if __name__ == "__main__":
    run_evidently_drift_reports()
