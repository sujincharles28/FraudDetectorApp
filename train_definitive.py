import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import numpy as np
import os

# --- 1. PREPROCESSING AND AUGMENTATION ---
def smart_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\+?\d[\d -]{8,12}\d', '[PHONE]', text)
    text = re.sub(r'\d+', '[NUMBER]', text)
    text = re.sub(r'[^a-z\s\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

new_fraud_examples = [
    "Immediate hiring for data entry clerk. Earn 3000 INR daily. No experience needed. A small refundable security deposit of 2000 INR is required for our portal access. Contact our HR on WhatsApp +919876543210.",
    "Work from home opportunity. Simple typing job. Get paid weekly. For registration, please send a security deposit of 1500 INR to our manager. Email us at jobs.now@gmail.com for details.",
    "Urgent requirement for online form filling. Basic computer knowledge is enough. We provide software but a one-time security deposit is mandatory. This is fully refundable. Apply via WhatsApp.",
    "Guaranteed income from home. Part-time job available. Earn 5000 INR daily. A security deposit for equipment is required before starting. No interview. Direct hiring.",
    "We are a global MNC looking for remote assistants. No experience required. High pay. To secure your spot, a refundable training fee of 3000 INR must be paid. Contact us at global.jobs.hr@gmail.com.",
    "Data entry operator needed. Simple copy paste work. Pay is 2500 INR daily. You must pay a security deposit for the software license. This is a one-time payment and is refunded after your first month.",
]

print("STEP 1: Loading, augmenting, and preprocessing data...")
df = pd.read_csv('fake_job_postings.csv')
# attach new fraud examples (same columns where possible)
new_data = pd.DataFrame({'full_text': new_fraud_examples, 'fraudulent': [1] * len(new_fraud_examples)})
for col in df.columns:
    if col not in ['full_text', 'fraudulent']:
        new_data[col] = ''
df = pd.concat([df, new_data], ignore_index=True)

df['clean_text'] = df['full_text'].apply(smart_clean_text)
df_model = df[['clean_text', 'fraudulent']].rename(columns={'fraudulent': 'label'})

# --- 2. SPLIT DATA (keep stratify) ---
print("STEP 2: Splitting data (stratified) and tokenizing...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_model['clean_text'], df_model['label'], test_size=0.2, random_state=42, stratify=df_model['label']
)

# compute class weights from training set (inverse frequency), clipped to avoid extreme weighting
train_label_counts = pd.Series(train_labels).value_counts().sort_index()
total = len(train_labels)
num_classes = len(train_label_counts)
# weight = total / (num_classes * count)
raw_weights = []
for i in range(num_classes):
    count = train_label_counts.get(i, 0)
    if count == 0:
        raw_weights.append(1.0)
    else:
        raw_weights.append(float(total) / (num_classes * count))
raw_weights = np.array(raw_weights, dtype=float)
# clip to conservative range [1.0, 4.0] — adjust upper bound if you want a bit more sensitivity
clipped = np.clip(raw_weights, a_min=1.0, a_max=4.0)
class_weights = torch.tensor(clipped, dtype=torch.float)
print(f"Computed class weights from training data (clipped): {class_weights}")

# --- 3. TOKENIZER AND DATASET PREP ---
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
special_tokens_dict = {'additional_special_tokens': ['[URL]', '[EMAIL]', '[PHONE]', '[NUMBER]']}
tokenizer.add_special_tokens(special_tokens_dict)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

class JobDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = JobDataset(train_encodings, list(train_labels))
val_dataset = JobDataset(val_encodings, list(val_labels))

# --- 4. METRICS FUNCTION ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1_fraudulent': f1, 'precision_fraudulent': precision, 'recall_fraudulent': recall}

# --- 5. CUSTOM WEIGHTED TRAINER ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 6. MODEL TRAINING SETUP ---
print("STEP 3: Loading model and preparing for final training...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.resize_token_embeddings(len(tokenizer))  # to include special tokens

# TrainingArguments tuned for a balanced objective
training_args = TrainingArguments(
    output_dir='./results_definitive',
    num_train_epochs=3,                   # small number of epochs with early stopping
    learning_rate=2e-5,                   # typical finetune LR for transformers
    per_device_train_batch_size=8,        # keep GPU memory in mind; effective batch = batch * accum
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,        # effective batch size multiplier
    warmup_ratio=0.06,                    # ~6% warmup
    weight_decay=0.01,
    logging_dir='./logs_definitive',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,                       # evaluate periodically (adjust depending on dataset size)
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1_fraudulent",    # must match compute_metrics key
    greater_is_better=True,
    fp16=True,                            # use mixed precision if available
    seed=42,
    dataloader_num_workers=2
)

# add EarlyStopping to avoid overfitting to extreme behaviors
callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

print("STEP 4: Starting final training...")
trainer.train()

# --- 7. POST-TRAINING: SAVE MODEL & TOKENIZER ---
save_path = './saved_model_definitive'
print("STEP 5: Saving model and tokenizer to", save_path)
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

# --- 8. THRESHOLD CALIBRATION (critical) ---
# We scan thresholds on the validation set and pick the threshold that maximizes F1 for fraud class.
print("STEP 6: Calibrating threshold on validation set to maximize F1 (fraud class)...")
predictions_output = trainer.predict(val_dataset)
logits = predictions_output.predictions
labels = predictions_output.label_ids
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]  # fraud probabilities

best_thresh = 0.5
best_f1 = -1.0
for t in np.linspace(0.05, 0.95, 19):
    preds_t = (probs >= t).astype(int)
    f1_t = f1_score(labels, preds_t, pos_label=1, zero_division=0)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = t

print(f"Best validation threshold: {best_thresh:.3f} with F1={best_f1:.4f}")

# save threshold for CLI to use
th_file = os.path.join(save_path, "best_threshold.txt")
with open(th_file, "w") as fh:
    fh.write(str(best_thresh))
print(f"Saved best threshold to {th_file}")

print("✅ Training + calibration complete. Model and threshold saved to './saved_model_definitive' ✅")
