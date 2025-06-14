import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# Charger le dataset
df = pd.read_csv('creditcard.csv')

# Afficher les premières lignes
print(df.head())

# Informations sur les colonnes et données
print(df.info())

# Vérifier la répartition des classes
print(df['Class'].value_counts())

# Visualiser la répartition classes (normal vs fraude)
sns.countplot(x='Class', data=df)
plt.title('Répartition transactions normales (0) et fraudes (1)')
plt.show()

# Séparer les données (features) et la cible (target)
X = df.drop('Class', axis=1)  # tout sauf la colonne Class
y = df['Class']               # seulement la colonne Class

# Diviser en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(random_state=42)

# Entraîner le modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Prédire sur les données test
y_pred = model.predict(X_test)
print("Modèle entraîné !")
# Afficher le rapport de classification (precision, recall, f1-score)
print(classification_report(y_test, y_pred))

# Afficher la matrice de confusion (vrai positif, faux positif, etc.)
print(confusion_matrix(y_test, y_pred))

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Créer un graphique
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraude'], 
            yticklabels=['Normal', 'Fraude'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()
