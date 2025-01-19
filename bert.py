import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Define Dataset Class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Load Training Data
def load_train_data(file_path):
    with open(file_path, "r") as f:
        train_data = json.load(f)
    texts, labels = [], []
    for label, sentences in train_data.items():
        texts.extend(sentences)
        labels.extend([label] * len(sentences))
    return texts, labels


# Load Annotated Test Data
def load_annotated_test_data(file_path):
    with open(file_path, "r") as f:
        annotated_data = json.load(f)
    texts, labels = [], []
    for label, sentences in annotated_data.items():
        texts.extend(sentences)
        labels.extend([label] * len(sentences))
    return texts, labels


# Load datasets
train_file = "./new_train.json"
annotated_test_file = "./annotated_dataset.json"

train_texts, train_labels = load_train_data(train_file)
test_texts, test_labels = load_annotated_test_data(annotated_test_file)

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    train_texts,
    train_labels_encoded,
    test_size=0.2,
    stratify=train_labels_encoded,
    random_state=42,
)

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_encoder.classes_)
)

# Prepare Dataloaders
train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_len=128)
val_dataset = TextClassificationDataset(X_val, y_val, tokenizer, max_len=128)
test_dataset = TextClassificationDataset(
    test_texts, test_labels_encoded, tokenizer, max_len=128
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Optimizer and Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()


# Training Function
def train_epoch(model, data_loader, optimizer, criterion, device):
    model = model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(
        data_loader
    )


# Evaluation Function
def evaluate_model(model, data_loader, criterion, device):
    model = model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(
        data_loader
    )


# Training Loop
num_epochs = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_acc, train_loss = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Evaluate on Validation Set
y_val_preds = []
y_val_true = []

model = model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        y_val_preds.extend(preds.cpu().numpy())
        y_val_true.extend(labels.cpu().numpy())

# Generate Classification Report for Validation Set
print(
    classification_report(y_val_true, y_val_preds, target_names=label_encoder.classes_)
)

# Confusion Matrix for Validation Set
cm_val = confusion_matrix(y_val_true, y_val_preds)
sns.heatmap(
    cm_val,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Validation Set Confusion Matrix")
plt.show()

########################### Evaluate on Test Set ###########################
y_test_preds = []
y_test_true = []

model = model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        y_test_preds.extend(preds.cpu().numpy())
        y_test_true.extend(labels.cpu().numpy())

# Generate Classification Report for Test Set
print(
    classification_report(
        y_test_true, y_test_preds, target_names=label_encoder.classes_
    )
)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test_true, y_test_preds)
sns.heatmap(
    cm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Test Set Confusion Matrix")
plt.show()
