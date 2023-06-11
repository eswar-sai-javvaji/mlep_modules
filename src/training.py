import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


logging.basicConfig(filename="model_log.log", filemode="w", level=logging.INFO)
logging.info("started training")

config = yaml.safe_load(open("config.yaml", "r"))
training_file = config["data_set_name"]["training_file"]
imputer_file = config["models"]["imputer_file"]
final_model_file = config["models"]["final_model_file"]
scaler_file = config["models"]["scaler_file"]
catencoder_file = config["models"]["catencoder_file"]
full_pipln_path = config["models"]["full_pipln_transform"]

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

data_prep_train = os.path.join("datasets/", "housing/", training_file)
strat_train_set = pd.read_csv(data_prep_train)
print(strat_train_set.shape)
logging.info("train data loaded")

housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set

housing_num = housing.drop("ocean_proximity", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()


imputer = SimpleImputer(strategy="median")

imputer_path = os.path.join("models", imputer_file)
pickle.dump(imputer, open(imputer_path, "wb"))

logging.info("imputer model is saved")

stnd_scaler = StandardScaler()

scaler_path = os.path.join("models", scaler_file)
pickle.dump(stnd_scaler, open(scaler_path, "wb"))

logging.info("scaler model is saved")

cat_encoder = OneHotEncoder()

catenc_path = os.path.join("models", catencoder_file)
pickle.dump(cat_encoder, open(catenc_path, "wb"))

logging.info("cat encoder model is saved")

attr_adder = CombinedAttributesAdder()

num_pipeline = Pipeline(
    [
        ("imputer", imputer),
        ("attribs_adder", attr_adder),
        ("std_scaler", stnd_scaler),
    ]
)


num_attribs = list(housing_num)  # only numeric columns
cat_attribs = ["ocean_proximity"]  # only categorical columns

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", cat_encoder, cat_attribs),
    ]
)

housing_prepared = full_pipeline.fit_transform(housing)

full_pipeline_pkl_path = os.path.join("models", full_pipln_path)
pickle.dump(full_pipeline, open(full_pipeline_pkl_path, "wb"))

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
    {"n_estimators": [30, 40, 50, 60], "max_features": [6, 8, 10, 12, 15]},
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

print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))


final_model = grid_search.best_estimator_

model_path = os.path.join("models", final_model_file)
pickle.dump(final_model, open(model_path, "wb"))

logging.info("saved the final model")
print("over")
logging.info("training completed")
