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

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de: {device}")

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
    
def prepare_data(train_path, test_path):
    print("Chargement des données...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['combined_text'] = train_df['Usage'] + " " + train_df['Text']
    test_df['combined_text'] = test_df['Usage'] + " " + test_df['Text']

    labels_to_ids = {label: idx for idx, label in enumerate(train_df['Label'].unique())}
    ids_to_labels = {idx: label for label, idx in labels_to_ids.items()}

    train_df['label_id'] = train_df['Label'].map(labels_to_ids)

    label_counts = Counter(train_df['label_id'])
    single_sample_classes = [label for label, count in label_counts.items() if count == 1]

    print(f"Nombre total de classes: {len(labels_to_ids)}")
    print(f"Nombre de classes avec un seul exemple: {len(single_sample_classes)}")

    if len(single_sample_classes) > 0:
        print("Détection de classes avec un seul exemple. Utilisation d'une division non stratifiée.")

        #Filtre des classes avec un seul exemple
        single_sample_indices = train_df[train_df['label_id'].isin(single_sample_classes)].index
        multi_sample_indices = train_df[~train_df['label_id'].isin(single_sample_classes)].index

        multi_sample_df = train_df.loc[multi_sample_indices]

        # Diviser les données avec stratification seulement pour les classes avec plusieurs exemples
        multi_train_idx, multi_val_idx = train_test_split(
            multi_sample_df.index,
            test_size=0.1,
            random_state=42,
            stratify=multi_sample_df['label_id']
        )

        #Ajouter les exemples uniques à l'ensemble d'entraînement
        train_indices = list(multi_train_idx) + list(single_sample_indices)
        val_indices = list(multi_val_idx)

        train_texts = train_df.loc[train_indices, 'combined_text'].values
        train_labels = train_df.loc[train_indices, 'label_id'].values
        val_texts = train_df.loc[val_indices, 'combined_text'].values
        val_labels = train_df.loc[val_indices, 'label_id'].values
    else:
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

    return (train_texts, val_texts, test_df['combined_text'].values, train_labels, val_labels, labels_to_ids, ids_to_labels, test_df)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Entraînement", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)

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

def generate_submission(model, test_dataloader, test_df, ids_to_labels, filename):
    print(f"Génération des prédictions pour le fichier {filename}...")
    _, test_predictions = evaluate(model, test_dataloader, device)
    
    test_pred_labels = [ids_to_labels[pred_id] for pred_id in test_predictions]
    
    submission_df = pd.DataFrame({
        'ID': test_df.index+1,
        'Label': test_pred_labels
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"Fichier de soumission '{filename}' créé avec succès.")

def main():
    TRAIN_BATCH_SIZE = 16  
    EVAL_BATCH_SIZE = 16  
    MAX_LENGTH = 128
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 0
    MODEL_NAME = "xlm-roberta-base"  
    TRAIN_PATH = "../data/train_submission.csv"
    TEST_PATH = "../data/test_without_labels.csv"

    train_texts, val_texts, test_texts, train_labels, val_labels, labels_to_ids, ids_to_labels, test_df = prepare_data(TRAIN_PATH, TEST_PATH)

    print(f"Chargement du tokenizer {MODEL_NAME}...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = LanguageDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer, max_length=MAX_LENGTH)
    val_dataset = LanguageDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer, max_length=MAX_LENGTH)
    test_dataset = LanguageDataset(texts=test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH)


    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2)

    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Chargement du modèle {MODEL_NAME}...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels_to_ids),
        problem_type="single_label_classification"
    )

    model.to(device)

    #optimiseur 
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_dataloader) * EPOCHS
    
    #scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        print(f"\nÉpoque {epoch + 1}/{EPOCHS}")

        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Perte moyenne d'entraînement: {avg_train_loss:.4f}")

        val_accuracy, _ = evaluate(model, val_dataloader, device)
        print(f"Précision de validation: {val_accuracy:.4f}")
        
        if epoch == 3:  
            print(f"Précision de validation à l'époque 4: {val_accuracy:.4f}")
            
            generate_submission(model, test_dataloader, test_df, ids_to_labels, "submission_epoch4.csv")

    print("\nSauvegarde du modèle final...")
    torch.save(model.state_dict(), "final_model.bin")
    print("Modèle final sauvegardé avec succès!")
    print(f"Précision finale de validation: {val_accuracy:.4f}")

    generate_submission(model, test_dataloader, test_df, ids_to_labels, "submission_final.csv")

if __name__ == "__main__":
    main()