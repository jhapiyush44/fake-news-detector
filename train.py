import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.metrics import accuracy_score


# =========================
# LOAD DATA
# =========================

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df])

df = df[["text", "label"]]

# Optional: reduce dataset size for faster training
df = df.sample(5000, random_state=42)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)


# =========================
# TOKENIZER
# =========================

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True
)


# =========================
# DATASET FORMAT
# =========================

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "label": list(train_labels)
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "label": list(test_labels)
})


# =========================
# MODEL
# =========================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }


# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10
)


# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# =========================
# TRAIN MODEL
# =========================

trainer.train()


# =========================
# SAVE MODEL
# =========================

model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Model training complete!")