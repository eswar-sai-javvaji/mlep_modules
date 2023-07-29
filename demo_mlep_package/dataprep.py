import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
import yaml
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

logging.basicConfig(filename="model_log.log", filemode="w", level=logging.INFO)
logging.info("started data preparation")


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

print(HOUSING_PATH)

flag = False
try:
    try:
        config = yaml.safe_load(open("config.yaml", "r"))
        flag = True
    except:
        print("config file is not available, so creating a new config file")
        logging.info(
            "config file is not available, so creating a new config file"
        )
        housing_file = "housing.csv"
        training_file = "train.csv"
        test_file = "test.csv"
        config = {
            "data_set_name": {
                "housing_file": None,
                "training_file": None,
                "test_file": None,
            },
        }
    if flag:
        housing_file = config["data_set_name"]["housing_file"]
        training_file = config["data_set_name"]["training_file"]
        test_file = config["data_set_name"]["test_file"]
        logging.info("opened the config file and extracted the values")
except:
    print(
        "data in config file is not correct so updating it with default values"
    )
    logging.info(
        "data in config file is not correct so updating it deafult values"
    )
    config = {
        "data_set_name": {
            "housing_file": None,
            "training_file": None,
            "test_file": None,
        },
    }
    housing_file = "housing.csv"
    training_file = "train.csv"
    test_file = "test.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--traindata", help="train data saving name", type=str)
parser.add_argument("--testdata", help="test data saving anme", type=str)
args_prsr = parser.parse_args()
# if args_prsr.housingdata:
#     housing_file = args_prsr.housingdata + ".csv"
if args_prsr.traindata:
    training_file = args_prsr.traindata + ".csv"
if args_prsr.testdata:
    test_file = args_prsr.testdata + ".csv"
print(housing_file)
print(training_file)
print(test_file)

logging.info("using argprase the argumets are taken")

config["data_set_name"]["housing_file"] = housing_file
config["data_set_name"]["training_file"] = training_file
config["data_set_name"]["test_file"] = test_file
with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
logging.info("updated the config file")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housingfile, housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, housingfile)
    return pd.read_csv(csv_path)


try:
    housing = load_housing_data(housing_file)
except:
    print("data is not available so downloading it")
    fetch_housing_data()
    housing = load_housing_data(housing_file)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
newdf = housing.select_dtypes(include=numerics)
corr_matrix = newdf.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)
housing["population_per_household"] = (
    housing["population"] / housing["households"]
)

logging.info("completed feature engineering")

print("test ", strat_test_set.shape)
print("train ", strat_train_set.shape)
print(HOUSING_PATH)
print(os.path.join(HOUSING_PATH, training_file))
print(os.path.join(HOUSING_PATH, test_file))
strat_test_set.to_csv(os.path.join(HOUSING_PATH, test_file), index=False)
strat_train_set.to_csv(os.path.join(HOUSING_PATH, training_file), index=False)
logging.info("train and test datasets are saved")
logging.info("data preparation completed")
