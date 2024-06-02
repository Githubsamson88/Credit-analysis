import warnings
import matplotlib
matplotlib.use('Agg')  # Configuration du backend non interactif pour Matplotlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, jaccard_score, hamming_loss
from catboost import CatBoostClassifier, Pool
import shap
import requests
from io import StringIO
import matplotlib.pyplot as plt
from ngboost import NGBClassifier  # Ajout de cette ligne

# Ignorer les avertissements PyplotGlobalUseWarning
warnings.filterwarnings("ignore", category=UserWarning, message="Matplotlib is currently using agg")

# Ignorer les avertissements PyplotGlobalUseWarning
warnings.filterwarnings("ignore", category=UserWarning, message="Matplotlib is currently using agg")

# URL du fichier CSV sur GitHub
github_url = 'https://raw.githubusercontent.com/Githubsamson88/Credit-analysis/main/data_M.csv'

# Téléchargement du fichier CSV depuis GitHub
response = requests.get(github_url)

# Lecture du contenu du fichier CSV
data = pd.read_csv(StringIO(response.text))

# Suppression de la colonne 'last_pymnt_d'
data.drop('last_pymnt_d', axis=1, inplace=True)

# Encodage des variables booléennes
bool_columns = data.select_dtypes(include=['bool']).columns
data[bool_columns] = data[bool_columns].astype(int)

# Suppression des pourcentages des colonnes int_rate et revol_util
if data['int_rate'].dtype == 'object':
    data['int_rate'] = data['int_rate'].str.replace('%', '')
    data['int_rate'] = pd.to_numeric(data['int_rate'], errors='coerce')

if data['revol_util'].dtype == 'object':
    data['revol_util'] = data['revol_util'].str.replace('%', '')
    data['revol_util'] = pd.to_numeric(data['revol_util'], errors='coerce')

# Conversion des colonnes catégorielles en type 'category'
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Suppression des valeurs manquantes
data = data.dropna()

# Encodage de loan_status
label_encoder = LabelEncoder()
data['loan_status'] = label_encoder.fit_transform(data['loan_status'])

# Définition des caractéristiques et de la cible
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des colonnes numériques
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int32', 'float32']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Création du modèle NGBoost
model_ngboost = NGBClassifier()

# Entraînement du modèle NGBoost
model_ngboost.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_ngboost = model_ngboost.predict(X_test)

# Calcul des métriques
accuracy_ngboost = accuracy_score(y_test, y_pred_ngboost)
jaccard_ngboost = jaccard_score(y_test, y_pred_ngboost, average='weighted')
hamming_ngboost = hamming_loss(y_test, y_pred_ngboost)

# Affichage des résultats
st.write("Résultats du modèle NGBoost :")
st.write("Accuracy :", accuracy_ngboost)
st.write("Jaccard Score :", jaccard_ngboost)
st.write("Hamming Loss :", hamming_ngboost)

# Visualisation 1 : data['annual_inc'].plot()
st.write("Visualisation 1 : Revenu annuel")
st.line_chart(data['annual_inc'])

# Visualisation 2 : Répartition des modalités de loan_status
st.write("Visualisation 2 : Répartition des modalités de loan_status")
total_samples = data.shape[0]
loan_status_percentages = data['loan_status'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
loan_status_percentages.plot(kind='bar', color=['blue', 'orange'])
plt.title('Répartition des modalités de loan_status')
plt.xlabel('loan_status')
plt.ylabel('Pourcentage')
plt.xticks(rotation=0)
for i, percentage in enumerate(loan_status_percentages):
    plt.text(i, percentage + 1, f'{percentage:.2f}%', ha='center')

plt.show()
st.pyplot()