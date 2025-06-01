import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os

# Load dataset
wine = load_wine()
x = wine.data
y = wine.target

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Set MLflow tracking URI and initialize DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Bhuvnn/ML-Flow-Practise.mlflow")
dagshub.init(repo_owner='Bhuvnn', repo_name='ML-Flow-Practise', mlflow=True)

# Set experiment and enable autologging
mlflow.set_experiment("MLOps GridSearchCV AutoLog Experiment")
mlflow.autolog()

# Define parameter grid
param_grid = {
    'max_depth': [5, 10, 15],
    'n_estimators': [10, 20, 50]
}

# Start MLflow run
with mlflow.start_run():
    # Apply GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(x_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_
    preds = best_model.predict(x_test)

    # Accuracy and confusion matrix
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix as artifact
    mlflow.log_artifact("confusion_matrix.png")

    # Optional: log script file
    if '__file__' in globals() and os.path.isfile(__file__):
        mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tags({"author": "Bhuvnn", "Project": "Wine Classification with GridSearchCV"})

    print("Best Params:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    print("Accuracy:", round(acc * 100, 2), "%")
