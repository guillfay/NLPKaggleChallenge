import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def train_language_classifier(csv_file):
    # Chargement des données depuis le CSV d'entraînement
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["Text", "Label"])
    X = df["Text"]
    y = df["Label"]

    # Séparation en 80% d'entraînement et 20% de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Création de la pipeline avec vectorisation TF-IDF et régression logistique
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 7), max_features=50000)),
            ("clf", LogisticRegression(max_iter=1000, solver="saga")),
        ]
    )

    # Définition de la grille d'hyperparamètres pour optimiser le modèle
    param_grid = {
        "tfidf__ngram_range": [(1, 2), (1, 3), (1, 4)],
        "tfidf__max_features": [10000, 50000, 100000],
        "clf__C": [0.1, 1, 10]
        }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("Meilleurs hyperparamètres :", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Évaluation sur l'ensemble de validation
    y_val_pred = best_model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print("===== Ensemble de Validation =====")
    print(classification_report(y_val, y_val_pred))
    print("Accuracy sur validation :", accuracy_val)

    return best_model


def predict_on_test(model, test_csv_file, output_csv_file="test_predictions.csv"):
    # Chargement des données de test (fichier sans labels)
    df_test = pd.read_csv(test_csv_file)
    # On suppose que le fichier de test possède une colonne "Text"
    X_test = df_test["Text"]

    # Prédiction sur l'ensemble de test
    predictions = model.predict(X_test)

    # Ajout des prédictions au dataframe
    df_test["Predicted_Label"] = predictions
    # Sauvegarde des résultats dans un nouveau CSV
    df_test.to_csv(output_csv_file, index=False)
    print(f"Les prédictions ont été sauvegardées dans {output_csv_file}")


if __name__ == "__main__":
    # Chemins vers vos fichiers CSV
    train_csv = "train_submission.csv"
    test_csv = "test_without_labels.csv"

    # Entraînement du modèle et optimisation des hyperparamètres sur l'ensemble d'entraînement
    best_model = train_language_classifier(train_csv)

    # Prédiction sur le CSV de test
    predict_on_test(best_model, test_csv)
