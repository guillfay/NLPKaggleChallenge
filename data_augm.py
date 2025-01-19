# Step 1: Data Augmentation with Paraphrasing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Load T5 model for paraphrasing
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Paraphrase function
def paraphrase_text(sentence, num_return_sequences=5):
    inputs = tokenizer(
        f"paraphrase: {sentence}", return_tensors="pt", max_length=256, truncation=True
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=256,
        num_return_sequences=num_return_sequences,
        num_beams=5,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Augment dataset using paraphrasing
def augment_dataset_with_paraphrasing(input_file, output_file, num_paraphrases=3):
    with open(input_file, "r") as f:
        train_data = json.load(f)

    augmented_data = {}
    for label, sentences in train_data.items():
        augmented_sentences = sentences[:]
        for sentence in sentences:
            paraphrased_sentences = paraphrase_text(
                sentence, num_return_sequences=num_paraphrases
            )
            augmented_sentences.extend(paraphrased_sentences)
        augmented_data[label] = augmented_sentences

    with open(output_file, "w") as f:
        json.dump(augmented_data, f, indent=4)


# File paths
input_file = "new_train.json"
output_file = "new_train_augmented.json"

# Augment data
augment_dataset_with_paraphrasing(input_file, output_file, num_paraphrases=2)
