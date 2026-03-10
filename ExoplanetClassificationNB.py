import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

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

# Limpieza de datos
features = [
    "Mass (MJ)",
    "Radius (RJ)",
    "Period (days)",
    "Temp. (K)",
    "Distance (ly)",
    "Host star mass (M☉)",
    "Host star temp. (K)"
]

for col in features:
    print(f"Tipo original de {col}:",data[col].dtype)
    # Todos eran str así que los cambiamos a un tipo de dato numerico
    # print(f"Cambiando {col} a un tipo de dato numerico...")
    data[col] = pd.to_numeric(data[col], errors='coerce')
    # print(f"Tipo original de {col}:",data[col].dtype)

data = data[features].copy()

# Eliminamos datos nulos
data = data.fillna(data.median())

# Distribucion de los datos
data.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

# Correlacion de las features
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# Escalar datos para clustering
scaler_cluster = StandardScaler()
scaled_features = scaler_cluster.fit_transform(data)

# Clustering para descubrir tipos de planeta
kmeans = KMeans(n_clusters=3, random_state=42)
data["PlanetType"] = kmeans.fit_predict(scaled_features)

print("Cluster distribution")
print(data["PlanetType"].value_counts())

centers = pd.DataFrame(
    scaler_cluster.inverse_transform(kmeans.cluster_centers_),
    columns=features
)

print("\nCluster Centers (interpretable planetary groups):")
print(centers)

# Uso del log-transform para variables muy sesgadas
'''
data["Mass (MJ)"] = np.log1p(data["Mass (MJ)"])
data["Distance (ly)"] = np.log1p(data["Distance (ly)"])
data["Period (days)"] = np.log1p(data["Period (days)"])
'''

X = data.drop(["PlanetType"], axis=1)
y = data["PlanetType"]

print(data.describe(percentiles=[0.1,0.9]))

print("Class distribution")
print(data["PlanetType"].value_counts(normalize=True))

# Train y test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Escalar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar al modelo
model = GaussianNB(priors=None)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print("\nNaive Bayes Results")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()