import warnings
import matplotlib
matplotlib.use('Agg')  # Configuration du backend non interactif pour Matplotlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier, Pool
import shap
import requests
from io import StringIO
import matplotlib.pyplot as plt

# Ignorer les avertissements PyplotGlobalUseWarning
warnings.filterwarnings("ignore", category=UserWarning, message="Matplotlib is currently using agg")

# Le reste de votre code...

# Création du modèle CatBoost
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, cat_features=categorical_features_indices, verbose=0)

# Entraînement du modèle
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100, plot=True)

# Prédiction
y_probs = model.predict_proba(X_test)[:, 1]

# Évaluation des performances du modèle
auc = roc_auc_score(y_test, y_probs)
st.write(f"Score AUC : {auc:.4f}")

# Création de l'explicateur SHAP pour CatBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap_values_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_train)

# Affichage du plot SHAP Summary
st.write("## SHAP Summary Plot")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
st.pyplot(fig_summary)

# Nettoyage de la figure
plt.close(fig_summary)

# Affichage du plot SHAP Waterfall
st.write("## SHAP Waterfall Plot")
shap.waterfall_plot(shap_values_exp[0])

st.pyplot()


# Tracer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
st.pyplot(bbox_inches='tight')
plt.clf()