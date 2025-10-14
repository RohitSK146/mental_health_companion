# preprocess_dataset.py
import pandas as pd

# Load data (expects data_train_raw.csv produced by prepare_dataset.py)
df = pd.read_csv("data_train_raw.csv")

# If label_name column exists (from corrected prepare script), use it. Otherwise try original label column.
if 'label_name' in df.columns:
    labels = df['label_name']
elif df['label'].dtype == 'int64' or df['label'].dtype == 'float64':
    # If only numeric labels present, keep them for now and raise a clear message
    raise ValueError("data_train_raw.csv contains numeric label ids but no 'label_name' column. "
                     "Please run the updated prepare_dataset.py that writes label names.")
else:
    labels = df['label']

# Mapping existing emotion labels to our 3-class setup
label_map = {
    "joy": "Positive",
    "love": "Positive",
    "surprise": "Positive",
    "sadness": "Negative",
    "anger": "Negative",
    "fear": "Risky",
    "Die":"Risky",
    "no problem":"Positive",
    "not bad":"Positive",
    "dont have any problem":"Positive"
}

# Apply mapping (works because we use label_name strings)
df['label'] = labels.map(label_map)

# Drop any unmapped rows (if exist) but print how many were dropped to help debug
before = len(df)
df = df.dropna(subset=['label'])
after = len(df)
dropped = before - after
print(f"Dropped {dropped} rows due to unmapped labels (if >0, check label names).")

# Keep only needed columns and save clean version
df_clean = df[['text','label']].reset_index(drop=True)
df_clean.to_csv("data_train_clean.csv", index=False)

print("✅ Cleaned dataset saved as data_train_clean.csv")
print(df_clean['label'].value_counts())
