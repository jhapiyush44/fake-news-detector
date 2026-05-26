import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# -----------------------------------
# Load Dataset
# -----------------------------------

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# -----------------------------------
# Create Better Input Text
# -----------------------------------

fake_df["content"] = fake_df["text"]
true_df["content"] = true_df["text"]

fake_df = fake_df[["content"]]
true_df = true_df[["content"]]

# -----------------------------------
# Labels
# 1 = Fake
# 0 = Real
# -----------------------------------

fake_df["label"] = 1
true_df["label"] = 0

# -----------------------------------
# Balanced Dataset
# -----------------------------------

fake_df = fake_df.sample(
    4000,
    random_state=42
)

true_df = true_df.sample(
    4000,
    random_state=42
)

# -----------------------------------
# Combine + Shuffle
# -----------------------------------

df = pd.concat([fake_df, true_df])

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# -----------------------------------
# Train / Validation Split
# -----------------------------------

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------------
# Model Name
# -----------------------------------

model_name = "roberta-base"

# -----------------------------------
# Load Tokenizer
# -----------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# -----------------------------------
# Tokenization Function
# -----------------------------------

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=384
    )

# -----------------------------------
# Create HuggingFace Datasets
# -----------------------------------

train_dataset = Dataset.from_dict({
    "text": train_texts.tolist(),
    "label": train_labels.tolist()
})

val_dataset = Dataset.from_dict({
    "text": val_texts.tolist(),
    "label": val_labels.tolist()
})

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# -----------------------------------
# Remove Text Column
# -----------------------------------

train_dataset = train_dataset.remove_columns(
    ["text"]
)

val_dataset = val_dataset.remove_columns(
    ["text"]
)

# -----------------------------------
# Set Torch Format
# -----------------------------------

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# -----------------------------------
# Load Model
# -----------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# -----------------------------------
# Metrics
# -----------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = torch.argmax(
        torch.tensor(logits),
        dim=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }

# -----------------------------------
# Training Arguments
# -----------------------------------

training_args = TrainingArguments(
    output_dir="./results",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    gradient_accumulation_steps=2,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    save_total_limit=1,

    fp16=True
)

# -----------------------------------
# Trainer
# -----------------------------------

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)

# -----------------------------------
# Train Model
# -----------------------------------

trainer.train()

# -----------------------------------
# Save Model + Tokenizer
# -----------------------------------

model.save_pretrained("model")

tokenizer.save_pretrained("model")

print("\nModel training complete!")
