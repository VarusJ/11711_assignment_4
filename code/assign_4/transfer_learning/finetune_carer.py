import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
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


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", problem_type="multi_label_classification")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_text(examples):
    return tokenizer(examples["text"], truncation=True)


def preprocess_label(examples):
    temp = np.zeros(len(label_list))
    temp[examples["labels"]] = 1.0
    return {"one_hot_labels": temp.tolist()}


dataset = load_dataset("dair-ai/emotion")
dataset = dataset.map(preprocess_text, batched=True)
dataset = dataset.map(preprocess_label)

label_list = list(set(dataset['train']['labels']))

dataset = dataset.rename_column("labels", "orginal_labels")
dataset = dataset.rename_column("one_hot_labels", "labels")

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]


id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred, threshold=0.3):
    pred, label = eval_pred
    pred = np.argmax(pred, axis=1)
    
    return {
        # "threshold": threshold,
        "accuracy": metrics.accuracy_score(label, pred),
        "precision": metrics.precision_score(label, pred, zero_division=0, average='macro'),
        "recall": metrics.recall_score(label, pred, zero_division=0, average='macro'),
        "f1": metrics.f1_score(label, pred, zero_division=0, average='macro'),
        # "mcc": metrics.matthews_corrcoef(label, pred),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="./bert_carer/results",
    logging_dir="./bert_carer/logs",
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
