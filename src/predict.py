import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# Configuración
EMBEDDINGS_FILE = '../results/vae_embeddings.tsv'
SURVIVAL_FILE = '../data/survival_data.tsv' 

def train_and_evaluate(X, y):
    """
    Replica la función 'trait_classification_accuracy' del estudio original.
    Usa Nested CV: Loop externo para evaluación (KFold), interno para ajustar hiperparámetros (GridSearchCV).
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    aucs = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Normalización (crucial para Logistic Regression y Neural Networks)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # GridSearch para optimizar 'C' (fuerza de regularización)
        # El estudio original usa penalización L1 (Lasso)
        params = {'C': [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000), 
                           params, cv=3, scoring='roc_auc')
        clf.fit(X_train, y_train)

        # Predicciones
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        accuracies.append(accuracy_score(y_test, preds))
        # Manejo de error si solo hay una clase en el test set
        try:
            aucs.append(roc_auc_score(y_test, probs))
        except ValueError:
            pass

    return np.mean(accuracies), np.mean(aucs)

# Cargar y alinear datos
X_df = pd.read_csv(EMBEDDINGS_FILE, sep='\t', index_col=0)
y_df = pd.read_csv(SURVIVAL_FILE, sep='\t', index_col=0)

# Intersección de pacientes comunes
common_patients = X_df.index.intersection(y_df.index)
X = X_df.loc[common_patients].values
# Crear etiqueta binaria: 1 si murió antes de 5 años (1825 días), 0 si vivió más
# Esto es una simplificación de la lógica de supervivencia del estudio original
y_data = y_df.loc[common_patients]
y_binary = ((y_data['fustat'] == 1) & (y_data['futime'] < 1825)).astype(int).values

print(f"Ejecutando predicción con {len(common_patients)} pacientes...")
acc, auc = train_and_evaluate(X, y_binary)
print(f"Resultados Promedio -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")