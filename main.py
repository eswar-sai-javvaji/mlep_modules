import os

print("*********started data prep*********")
os.system("python src/data_pep.py")
print("*********data preparation completed*********")

print("*********started training*********")
os.system("python src/training.py")
print("*********training completed*********")

print("*********started scoring*********")
os.system("python src/scoring.py")
print("*********scoring completed*********")
