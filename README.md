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
conda create --name mle-dev
installed nessesary packages using pip as mentioned in env file
conda actiavte mle-dev
The script is now divided into 3 parts data_pep training and scoring
i have added a config file which had the names of the data files and models saved
we need to provide the names for train and test datasets and pikel models which are saved
we can provide them via config file or through comand propt as arguments
then run the script in the above mentioned order by giving specific arguments