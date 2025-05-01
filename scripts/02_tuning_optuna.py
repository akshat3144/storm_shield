import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load your dataset
def load_data():
    data = load_iris()  # Replace with your dataset
    X, y = data.data, data.target
    return X, y

# Define the objective function
def objective(trial):
    X, y = load_data()

    # Example hyperparameters for RandomForest
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    # Define model pipeline (with optional scaler)
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score

# Run the optimization
def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial)
    print("Best parameters:")
    print(study.best_params)

if __name__ == "__main__":
    main()
