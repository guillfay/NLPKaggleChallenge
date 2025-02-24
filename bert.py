import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Chargement et préparation des données
df = pd.read_csv("train_submission.csv")
df = df.dropna(subset=["Text", "Label"])

# Création des mappings label -> id
labels = df["Label"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["Label"].map(label2id)

# 2. Séparation en 80% train / 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Conversion en datasets Hugging Face
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 3. Tokenisation
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(examples):
    return tokenizer(
        examples["Text"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# On précise les colonnes à conserver pour le modèle
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 4. Chargement du modèle pour la classification
num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
)


# 5. Définition des métriques
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# 6. Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Lancement du fine-tuning
trainer.train()

# Évaluation sur l'ensemble de validation
eval_results = trainer.evaluate()
print("Résultats sur l'ensemble de validation :", eval_results)

# Optionnel : Sauvegarde du modèle fine-tuné
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")
# Chargement du CSV de test
df_test = pd.read_csv("test_without_labels.csv")
test_dataset = Dataset.from_pandas(df_test)

# Tokenisation
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Prédiction
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

# Ajout des prédictions dans le dataframe
df_test["Predicted_Label"] = [id2label[p] for p in preds]
df_test.to_csv("test_predictions_bert.csv", index=False)
print("Les prédictions ont été sauvegardées dans test_predictions_bert.csv")
