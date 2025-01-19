import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load NLTK resources
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Initialize global tools
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    text = " ".join(
        [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(text)
            if word not in stop_words
        ]
    )
    return text


# Load training data
def load_train_data(file_path):
    with open(file_path, "r") as f:
        train_data = json.load(f)
    texts, labels = [], []
    for label, sentences in train_data.items():
        texts.extend(sentences)
        labels.extend([label] * len(sentences))
    df = pd.DataFrame({"text": texts, "label": labels})
    df["text"] = df["text"].apply(preprocess_text)
    return df


# Load test data
def load_test_data(file_path):
    with open(file_path, "r") as f:
        texts = f.readlines()
    df = pd.DataFrame({"text": [line.strip() for line in texts]})
    df["text"] = df["text"].apply(preprocess_text)
    return df


# Load datasets
train_file = "./new_train.json"
test_file = "./test_shuffle.txt"

train_df = load_train_data(train_file)
test_df = load_test_data(test_file)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Add bigram features
X_train = tfidf.fit_transform(train_df["text"]).toarray()
y_train = train_df["label"]
X_test = tfidf.transform(test_df["text"]).toarray()

# Train-test split for evaluation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_split_encoded = label_encoder.transform(y_train_split)
y_val_split_encoded = label_encoder.transform(y_val_split)

# Train the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
)

xgb_model.fit(X_train_split, y_train_split_encoded)

# Predict on validation set
y_val_pred_encoded = xgb_model.predict(X_val_split)

# Decode predictions back to original labels
y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

# Evaluation
print("Validation Metrics:")
print(confusion_matrix(y_val_split, y_val_pred))
print(classification_report(y_val_split, y_val_pred))


# Visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


plot_confusion_matrix(y_val_split, y_val_pred, classes=train_df["label"].unique())

# Predict on test set
#y_test_pred = xgb_model.predict(X_test)
#test_df["predicted_label"] = y_test_pred
#test_df[["text", "predicted_label"]].to_csv("predictions_xgb.csv", index=False)