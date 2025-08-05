# run.py
# This script handles training and evaluation for the Iris Decision Tree model,
# with integration for MLflow experiment tracking.

import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import MLflow and MLflow's scikit-learn integration
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def run(mode):
    """
    Runs the specified mode: 'train' or 'evaluate'.
    """
    # --- Shared Code: Load and Split Data ---
    print("Loading data from 'data/iris.csv'...")
    data = pd.read_csv('data/iris.csv')

    print("Splitting data...")
    train_df, test_df = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)

    X_train = train_df[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train_df.species
    X_test = test_df[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test_df.species

    # --- Mode-specific logic ---
    if mode == 'train':
        print("--- Running in TRAIN mode with MLflow tracking ---")

        mlflow.set_experiment("IRIS_Classifier_Pipeline")

        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")

            # --- Model Training ---
            max_depth_param = 3
            random_state_param = 1
            
            print("Logging parameters to MLflow...")
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            mlflow.log_param("max_depth", max_depth_param)
            mlflow.log_param("random_state", random_state_param)

            print("Training Decision Tree model...")
            model = DecisionTreeClassifier(max_depth=max_depth_param, random_state=random_state_param)
            model.fit(X_train, y_train)

            # --- Evaluation & Metric Logging ---
            print("Evaluating the model...")
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            print("Logging metrics to MLflow...")
            mlflow.log_metric("accuracy", accuracy)
            
            # --- Artifact Logging (Report and Model) ---
            print("Logging artifacts to MLflow...")
            report_text = classification_report(y_test, predictions)
            mlflow.log_text(report_text, "classification_report.txt")

            # Infer the model signature and provide an input example
            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_train.head(1)

            # Log the model with its signature and an example
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="iris_decision_tree_model",
                signature=signature,
                input_example=input_example
            )
            
            # Save a local copy for subsequent CI steps
            os.makedirs('artifacts', exist_ok=True)
            joblib.dump(model, 'artifacts/model.joblib')
            
            print("Training and MLflow logging complete.")
            print(f"🏃 View run at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

    elif mode == 'evaluate':
        print("--- Running in EVALUATE mode (for CML report) ---")
        model = joblib.load('artifacts/model.joblib')
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report_text = classification_report(y_test, predictions)

        # Print metrics in Markdown format for the CML report
        print(f'**Accuracy:** {accuracy:.3f}')
        print('')
        print('**Classification Report:**')
        print('```')
        print(report_text)
        print('```')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training or evaluation for the Iris model with MLflow.")
    parser.add_argument('mode', choices=['train', 'evaluate'], help="The mode to run: 'train' or 'evaluate'")
    args = parser.parse_args()
    run(args.mode)
