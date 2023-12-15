import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from sklearn import metrics


label_list = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

tokenizer = AutoTokenizer.from_pretrained("/home/zjing2/capstone/goemotions/bert_carer/results/checkpoint-5000", problem_type="multi_label_classification")

import pandas as pd


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_text(examples):
    return tokenizer(examples["text"], truncation=True)


def preprocess_label(examples):
    temp = np.zeros(len(label_list))
    temp[examples["labels"]] = 1.0
    return {"one_hot_labels": temp.tolist()}


dataset = load_dataset("go_emotions")
df = pd.read_csv("/home/zjing2/capstone/goemotions/train_underperform_BERTaugmented.tsv", sep='\t', header=None, names=['text', 'labels'])
df["labels"] = df["labels"].apply(lambda x: [int(i) for i in x.split(',')])
df['id'] = list(range(len(df)))
train_set = Dataset.from_pandas(df)

dataset = dataset.map(preprocess_text, batched=True)
dataset = dataset.map(preprocess_label)
train_set = train_set.map(preprocess_text, batched=True)
train_set = train_set.map(preprocess_label)

dataset = dataset.rename_column("labels", "orginal_labels")
dataset = dataset.rename_column("one_hot_labels", "labels")
train_set = train_set.rename_column("labels", "orginal_labels")
train_set = train_set.rename_column("one_hot_labels", "labels")

train_dataset = train_set
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


model = AutoModelForSequenceClassification.from_pretrained(
    "/home/zjing2/capstone/goemotions/bert_carer/results/checkpoint-5000",
    problem_type="multi_label_classification",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred, threshold=0.3):
    predictions = eval_pred.predictions
    label = eval_pred.label_ids
    pred = torch.nn.functional.softmax(torch.Tensor(predictions), dim=1)
    pred = (pred>threshold)*1.0
    
    return {
        "threshold": threshold,
        "accuracy": metrics.accuracy_score(label, pred),
        "precision": metrics.precision_score(label, pred, zero_division=0, average='macro'),
        "recall": metrics.recall_score(label, pred, zero_division=0, average='macro'),
        "f1": metrics.f1_score(label, pred, zero_division=0, average='macro'),
    }



training_args = TrainingArguments(
    output_dir="./bert_carer_minor_bert/results",
    logging_dir="./bert_carer_minor_bert/logs",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=11711
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.evaluate()
