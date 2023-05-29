import logging
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error

logging.basicConfig(filename="model_log.log", filemode="w", level=logging.INFO)
logging.info("started scoring2")

config = yaml.safe_load(open("config.yaml", "r"))

test_file = config["data_set_name"]["test_file"]
imputer_file = config["models"]["imputer_file"]
final_model_file = config["models"]["final_model_file"]

data_prep_test = os.path.join("datasets/", "housing/", test_file)
strat_test_set = pd.read_csv(data_prep_test)
print(strat_test_set.shape)
logging.info("test data loaded")

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)

imputer_path = os.path.join("models", imputer_file)
imputer = pickle.load(open(imputer_path, "rb"))

logging.info("loaded the imputer model from pikel files")

X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True)
)

model_path = os.path.join("models", final_model_file)
final_model = pickle.load(open(model_path, "rb"))

logging.info("loaded the final model from pikel files")

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

logging.info("forecasting is completed")
print(final_rmse)
logging.info("scoring completed")
