"""
Análise de Sentimentos de Tweets usando:
Modelo Hugging Face:
neuralmind/bert-base-portuguese-cased

Objetivo:
- Classificar tweets em:
    positivo
    negativo
    neutro

Observação:
O modelo "neuralmind/bert-base-portuguese-cased"
é um BERT base pré-treinado em português.
Ele NÃO vem ajustado para sentimentos.

Por isso, este exemplo:
1. Usa o BERT como encoder
2. Adiciona uma camada de classificação
3. Permite treinar com seus próprios dados

Requisitos:
pip install transformers datasets torch pandas scikit-learn accelerate

GPU CUDA:
O código usa GPU automaticamente se disponível.
"""

from warnings import filterwarnings

filterwarnings("ignore")

from typing import Dict

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# =========================================================
# CONFIGURAÇÕES
# =========================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

OUTPUT_DIR = "./modelo_sentimentos"

# =========================================================
# LABELS
# =========================================================

label2id = {
    "negativo": 0,
    "neutro": 1,
    "positivo": 2,
}

id2label = {
    0: "negativo",
    1: "neutro",
    2: "positivo",
}

# =========================================================
# EXEMPLO DE DATASET
# Substitua pelo seu CSV de tweets
# =========================================================

dados = {
    "tweet": [
        "Esse filme é maravilhoso!",
        "O atendimento foi horrível.",
        "Hoje o dia está normal.",
        "Gostei muito desse celular.",
        "Que experiência péssima.",
        "Nada demais aconteceu hoje.",
        "Excelente trabalho da equipe.",
        "Estou decepcionado com o produto.",
        "Foi um evento ok.",
    ],
    "sentimento": [
        "positivo",
        "negativo",
        "neutro",
        "positivo",
        "negativo",
        "neutro",
        "positivo",
        "negativo",
        "neutro",
    ],
}

df = pd.DataFrame(dados)
df = pd.read_csv("tweets_sentimentos.csv", sep=",", encoding="utf-8")

# =========================================================
# CONVERTE LABELS
# =========================================================

df["label"] = df["sentimento"].map(label2id)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./cache")


def tokenize_function(examples: Dict):
    return tokenizer(
        examples["content"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# =========================================================
# REMOVE COLUNAS DESNECESSÁRIAS
# =========================================================

columns_to_remove = [
    col
    for col in train_dataset.column_names
    if col not in ["input_ids", "attention_mask", "label"]
]

train_dataset = train_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)

# =========================================================
# FORMATO TORCH
# =========================================================

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# =========================================================
# MODELO
# =========================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    cache_dir="./cache",
)

# =========================================================
# GPU
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"\nUsando dispositivo: {device}")

# =========================================================
# MÉTRICAS
# =========================================================


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1,
    }


# =========================================================
# DATA COLLATOR
# =========================================================

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =========================================================
# TREINAMENTO
# =========================================================

print("\nIniciando treinamento...\n")
trainer.train()

# =========================================================
# AVALIAÇÃO
# =========================================================

print("\nAvaliação do modelo:\n")

metrics = trainer.evaluate()

print(metrics)

# =========================================================
# SALVAR MODELO
# =========================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModelo salvo em: {OUTPUT_DIR}")

# =========================================================
# FUNÇÃO DE PREDIÇÃO
# =========================================================


def prever_sentimento(texto: str):
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=-1).item()

    return id2label[prediction]


# =========================================================
# TESTE
# =========================================================

frases = [
    "Esse notebook é excelente!",
    "O serviço foi muito ruim.",
    "Hoje fui ao mercado.",
]

print("\nPredições:\n")

for frase in frases:
    sentimento = prever_sentimento(frase)

    print(f"Texto: {frase}")
    print(f"Sentimento: {sentimento}")
    print("-" * 50)
