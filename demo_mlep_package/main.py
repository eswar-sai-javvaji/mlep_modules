import os

import demo_mlep_package

print("*********started data prep*********")
os.system("python -m demo_mlep_package.dataprep")
print("*********data preparation completed*********")

print("*********started training*********")
os.system("python -m demo_mlep_package.training")
print("*********training completed*********")

print("*********started scoring*********")
os.system("python -m demo_mlep_package.scoring")
print("*********scoring completed*********")
