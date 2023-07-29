# mle training
# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
#### conda env
1. conda create --name mle-dev
2. installed nessesary packages using pip as mentioned in env file
3. conda actiavte mle-dev
4. The script is now divided into 3 parts dataprep, training and scoring
5. I have added a config file which had the names of the data files and models saved
6. we need to provide the names for train and test datasets and pikel models which are going to be saved
7. we can provide them via config file or through comand propt as arguments
8. then run the script in the above mentioned order by giving specific arguments
9. the code will update the arguments in config and takes from there, addition to that it will log the details of execution in model-log file for every script run

# To run directly using mlflow
Created a script called main.py in src folder - please run that script which automaticaly runs all src scripts
also created a mlproject file for mlflow and made code in a single package which can run by mlflow
please use command "mlflow run . " to run the whole project with help of mlflow
please use the command "mlflow ui" to see the ui in you localhost - "localhost:5000"

# To run package
Create an environment using env.yml
Please install the .whl created from packaging project file in dist folder using pip
run the python module - "python -m demo_mlep_package.main"
this command will execute the codes and creates the config files needed and logs the info in log file and saves the pkl files in models folder
## Note for package:
if the data url woudnt work please paste the data 'housing.csv' provided with this files in a way - 'dataset/housig/housing.csv'