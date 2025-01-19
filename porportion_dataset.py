import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Load NLTK resources
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Initialize tools
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

    # Check text before and after preprocessing
    df["clean_text"] = df["text"].apply(preprocess_text)
    print("\nSample Text Comparison:")
    for i in range(3):  # Show a few examples
        print(f"Original: {df['text'].iloc[i]}")
        print(f"Preprocessed: {df['clean_text'].iloc[i]}\n")

    return df


# Dataset size and class distribution check
def analyze_dataset_distribution(df, title):
    print(f"\n{title}:")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())
    sns.countplot(x=df["label"])
    plt.title(f"{title} Class Distribution")
    plt.xticks(rotation=90)
    plt.show()


# Load datasets
train_file = "./new_train.json"
train_df = load_train_data(train_file)

# Analyze dataset distribution
analyze_dataset_distribution(train_df, "Training Dataset")

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Add bigram features
X_train = tfidf.fit_transform(train_df["clean_text"]).toarray()
y_train = train_df["label"]

# Split data with stratification
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Analyze validation set distribution
val_df = pd.DataFrame({"text": X_val_split.tolist(), "label": y_val_split})
analyze_dataset_distribution(val_df, "Validation Dataset")
