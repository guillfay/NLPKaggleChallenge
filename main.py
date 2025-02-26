import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os
import random
from collections import Counter

# Définir des seed pour la reproductibilité
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# Vérifier si GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de: {device}")

# Dataset personnalisé pour les textes multilingues
class LanguageDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tronquer le texte si trop long (pour éviter les problèmes de mémoire)
        if len(text) > 1024:
            text = text[:1024]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        outputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            outputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return outputs

# Fonction pour charger et préparer les données
def prepare_data(train_path, test_path):
    print("Chargement des données...")

    # Charger les données d'entraînement et de test
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Créer une colonne 'text' combinant Usage et Text pour une meilleure classification
    train_df['combined_text'] = train_df['Usage'] + " " + train_df['Text']
    test_df['combined_text'] = test_df['Usage'] + " " + test_df['Text']

    # Map les labels en entiers
    labels_to_ids = {label: idx for idx, label in enumerate(train_df['Label'].unique())}
    ids_to_labels = {idx: label for label, idx in labels_to_ids.items()}

    # Conversion des labels en IDs
    train_df['label_id'] = train_df['Label'].map(labels_to_ids)

    # Analyser la distribution des classes
    label_counts = Counter(train_df['label_id'])
    single_sample_classes = [label for label, count in label_counts.items() if count == 1]

    # Afficher des statistiques sur les classes à faible représentation
    print(f"Nombre total de classes: {len(labels_to_ids)}")
    print(f"Nombre de classes avec un seul exemple: {len(single_sample_classes)}")

    # Pour les classes avec un seul exemple, nous ne pouvons pas les stratifier
    # Option 1: Créer une validation fixe pour ces classes
    # Option 2: Les garder uniquement dans l'ensemble d'entraînement

    # Créer ensembles d'entraînement et validation sans stratification pour éviter l'erreur
    if len(single_sample_classes) > 0:
        print("Détection de classes avec un seul exemple. Utilisation d'une division non stratifiée.")

        # Filtrer les classes avec un seul exemple
        single_sample_indices = train_df[train_df['label_id'].isin(single_sample_classes)].index
        multi_sample_indices = train_df[~train_df['label_id'].isin(single_sample_classes)].index

        # Dataset pour les classes avec plusieurs exemples
        multi_sample_df = train_df.loc[multi_sample_indices]

        # Diviser les données avec stratification seulement pour les classes avec plusieurs exemples
        multi_train_idx, multi_val_idx = train_test_split(
            multi_sample_df.index,
            test_size=0.1,
            random_state=42,
            stratify=multi_sample_df['label_id']
        )

        # Ajouter les exemples uniques à l'ensemble d'entraînement
        train_indices = list(multi_train_idx) + list(single_sample_indices)
        val_indices = list(multi_val_idx)

        # Créer les ensembles finaux
        train_texts = train_df.loc[train_indices, 'combined_text'].values
        train_labels = train_df.loc[train_indices, 'label_id'].values
        val_texts = train_df.loc[val_indices, 'combined_text'].values
        val_labels = train_df.loc[val_indices, 'label_id'].values
    else:
        # Division standard avec stratification
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_df['combined_text'].values,
            train_df['label_id'].values,
            test_size=0.1,
            random_state=42,
            stratify=train_df['label_id']
        )

    print(f"Nombre d'exemples d'entraînement: {len(train_texts)}")
    print(f"Nombre d'exemples de validation: {len(val_texts)}")
    print(f"Nombre d'exemples de test: {len(test_df)}")

    return (train_texts, val_texts, test_df['combined_text'].values,
            train_labels, val_labels, labels_to_ids, ids_to_labels)

# Fonction d'entraînement
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Entraînement", leave=True)

    for batch in progress_bar:
        # Déplacer les données sur le dispositif approprié
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Mise à jour de la barre de progression
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)

# Fonction d'évaluation
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Évaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())

            if 'labels' in batch:
                labels = batch['labels'].to(device)
                actual_labels.extend(labels.cpu().tolist())

    if actual_labels:
        accuracy = accuracy_score(actual_labels, predictions)
        return accuracy, predictions

    return None, predictions

# Fonction principale
def main():
    # Paramètres
    TRAIN_BATCH_SIZE = 16  # Réduit pour Colab
    EVAL_BATCH_SIZE = 16  # Réduit pour Colab
    MAX_LENGTH = 128
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 0
    MODEL_NAME = "xlm-roberta-base"  # Modèle pré-entraîné multilingue

    # Chemins des fichiers
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"

    # Préparer les données
    train_texts, val_texts, test_texts, train_labels, val_labels, labels_to_ids, ids_to_labels = prepare_data(
        TRAIN_PATH, TEST_PATH
    )

    # Initialiser le tokenizer
    print(f"Chargement du tokenizer {MODEL_NAME}...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # Créer les datasets
    train_dataset = LanguageDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    val_dataset = LanguageDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    test_dataset = LanguageDataset(
        texts=test_texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    # Créer les dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Initialiser le modèle
    print(f"Chargement du modèle {MODEL_NAME}...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels_to_ids),
        problem_type="single_label_classification"
    )

    model.to(device)

    # Initialiser l'optimiseur
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Nombre total d'étapes d'entraînement
    total_steps = len(train_dataloader) * EPOCHS

    # Créer le scheduler pour diminuer le taux d'apprentissage
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Entraînement
    print("Début de l'entraînement...")
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"\nÉpoque {epoch + 1}/{EPOCHS}")

        # Entraînement
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Perte moyenne d'entraînement: {avg_train_loss:.4f}")

        # Évaluation
        val_accuracy, _ = evaluate(model, val_dataloader, device)
        print(f"Précision de validation: {val_accuracy:.4f}")

        # Conserver uniquement le meilleur modèle
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.bin")
            print("Meilleur modèle sauvegardé!")

    # Sauvegarder le modèle final
    print("\nSauvegarde du modèle final...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }, "final_model.pt")

    # Charger le meilleur modèle pour les prédictions
    print("\nChargement du meilleur modèle pour les prédictions...")
    model.load_state_dict(torch.load("best_model.bin"))

    # Prédictions sur l'ensemble de test
    print("Génération des prédictions pour l'ensemble de test...")
    _, test_predictions = evaluate(model, test_dataloader, device)

    # Convertir les IDs de prédiction en labels
    test_pred_labels = [ids_to_labels[pred_id] for pred_id in test_predictions]

    # Créer le fichier de soumission
    test_df = pd.read_csv(TEST_PATH)
    submission_df = pd.DataFrame({
        'id': test_df.index,
        'prediction': test_pred_labels
    })

    submission_df.to_csv('submission.csv', index=False)
    print("Fichier de soumission 'submission.csv' créé avec succès.")

if __name__ == "__main__":
    main()