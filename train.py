import os
import pickle
import click
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    """Load a pickled object from a file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # Load preprocessed data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # Initialize RandomForestRegressor
    rf = RandomForestRegressor(random_state=0)
    
    # Check and print the current value of min_samples_split
    print("Current min_samples_split:", rf.get_params()['min_samples_split'])

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on validation data
    y_pred = rf.predict(X_val)

    # Calculate RMSE
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print("RMSE on validation set:", rmse)

    # GridSearchCV to find the best min_samples_split
    param_grid = {
        'min_samples_split': [2, 4, 8, 10]
    }

    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best value of min_samples_split from GridSearchCV
    print("Best min_samples_split from GridSearchCV:", grid_search.best_params_['min_samples_split'])

    # You can also re-train the model using the best hyperparameter:
    best_rf = RandomForestRegressor(min_samples_split=grid_search.best_params_['min_samples_split'], random_state=0)
    best_rf.fit(X_train, y_train)
    y_pred_best = best_rf.predict(X_val)
    rmse_best = mean_squared_error(y_val, y_pred_best, squared=False)
    print("RMSE on validation set with best min_samples_split:", rmse_best)

if __name__ == '__main__':
    run_train()
