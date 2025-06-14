# Détection de Fraude Bancaire avec Machine Learning

## 📌 Contexte

Dans un monde bancaire de plus en plus digitalisé, les fraudes par carte bancaire deviennent fréquentes et difficiles à détecter en temps réel.  
Ce projet a pour but de créer un modèle de machine learning capable de détecter automatiquement les transactions frauduleuses en se basant sur des données historiques.

---

## ⚙️ Description du Projet

Ce projet utilise un modèle de **Random Forest** entraîné sur le dataset `creditcard.csv`, un jeu de données réel contenant des transactions (normales et frauduleuses).  
L'objectif est de **prédire si une transaction est frauduleuse** à partir de ses caractéristiques.

---

## 📁 Fichiers du Projet

- `fraude.py` : script principal avec le code complet (préparation, entraînement, évaluation)
- `README.md` : ce fichier
- (Pas de dataset dans le dépôt pour des raisons de taille)

---

## 📊 Données

Le dataset utilisé provient de **Kaggle** :  
👉 [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**⚠️ Important :** Téléchargez manuellement le fichier `creditcard.csv` depuis Kaggle et placez-le dans le même dossier que le script `fraude.py`.

---

## ✅ Résultats du modèle

- Précision sur la classe 1 (fraude) : 94%
- Recall sur la fraude : 82%
- F1-score global : 0.87

---

## 🔧 Librairies utilisées

- pandas
- matplotlib
- seaborn
- scikit-learn

---

## 📸 Visualisation

Quelques graphes exploratoires ont été réalisés (répartition des classes, etc.).

---

## 💡 Idées d’amélioration

- Test d'autres modèles : XGBoost, LightGBM
- Undersampling / Oversampling pour l’équilibrage
- Cross-validation
