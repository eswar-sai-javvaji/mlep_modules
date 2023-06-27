import logging
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

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
logging.info("started scoring2")

config = yaml.safe_load(open("config.yaml", "r"))

test_file = config["data_set_name"]["test_file"]
imputer_file = config["models"]["imputer_file"]
final_model_file = config["models"]["final_model_file"]
scaler_file = config["models"]["scaler_file"]
catencoder_file = config["models"]["catencoder_file"]
full_pipln_path = config["models"]["full_pipln_transform"]


data_prep_test = os.path.join("datasets/", "housing/", test_file)
strat_test_set = pd.read_csv(data_prep_test)
print(strat_test_set.shape)
logging.info("test data loaded")

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()


X_test_num = X_test.drop("ocean_proximity", axis=1)

full_pipeline_pkl_path = os.path.join("models", full_pipln_path)
full_pipeline = pickle.load(open(full_pipeline_pkl_path, "rb"))
logging.info("loaded the encoder & tranfrom comb model from pikel files")

attr_adder = CombinedAttributesAdder()

test_prepared = full_pipeline.transform(X_test)

print(pd.DataFrame(test_prepared).head())

model_path = os.path.join("models", final_model_file)
final_model = pickle.load(open(model_path, "rb"))

logging.info("loaded the final model from pikel files")

final_predictions = final_model.predict(test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

logging.info("forecasting is completed")
print(final_rmse)
logging.info("scoring completed")
print(final_model)
print(final_model.get_params())
