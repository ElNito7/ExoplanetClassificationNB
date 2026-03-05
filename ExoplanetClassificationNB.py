import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# Descargar dataset
path = kagglehub.dataset_download("diaaessam/exoplanets-planets-outside-our-galaxy")

print("Path to dataset files:", path)
print(os.listdir(path))

# Leer CSV
file_path = os.path.join(path, "exoplanets.csv")
data = pd.read_csv(file_path)

# Ver información básica
pd.set_option('display.max_columns', None)
print(" Data info: ")
data.info()
print(" Describe data: ")
data.describe()
print(" Missing values per column:")
print(data.isnull().sum())
print("Dta head ")
print(data.head())
data.isnull().sum()  # Check for missing values