import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

# import tarfile

logging.basicConfig(filename="model_log.log", filemode="w", level=logging.INFO)
logging.info("started training")

config = yaml.safe_load(open("config.yaml", "r"))
training_file = config["data_set_name"]["training_file"]
imputer_file = config["models"]["imputer_file"]
final_model_file = config["models"]["final_model_file"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--imputermodel", help="imputer model saving name", type=str
)
parser.add_argument("--finalmodel", help="final model saving anme", type=str)
args_prsr = parser.parse_args()

if args_prsr.imputermodel:
    imputer_file = args_prsr.imputermodel + ".pkl"
if args_prsr.finalmodel:
    final_model_file = args_prsr.finalmodel + ".pkl"

config["models"]["imputer_file"] = imputer_file
config["models"]["final_model_file"] = final_model_file
with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

data_prep_train = os.path.join("../datasets/", "housing/", training_file)
strat_train_set = pd.read_csv(data_prep_train)
print(strat_train_set.shape)
logging.info("train data loaded")

housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

imputer_path = os.path.join("models", imputer_file)
pickle.dump(imputer, open(imputer_path, "wb"))

logging.info("imputer model is saved")

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr["rooms_per_household"] = (
    housing_tr["total_rooms"] / housing_tr["households"]
)
housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(
    pd.get_dummies(housing_cat, drop_first=True)
)

logging.info("started training different models")
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

print("linear", lin_rmse)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

print("tree", tree_rmse)

param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)


final_model = grid_search.best_estimator_

model_path = os.path.join("models", final_model_file)
pickle.dump(lin_reg, open(model_path, "wb"))

logging.info("saved the final model")
print("over")
logging.info("training completed")
