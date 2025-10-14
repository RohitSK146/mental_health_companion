# prepare_dataset.py
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Get label names mapping (list indexed by label id)
label_names = dataset['train'].features['label'].names  # e.g., ["joy", "sadness", ...]

# Convert to pandas DataFrame with label names
train = dataset['train'] 
test = dataset['test']

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

# Convert numeric label -> label name for readability/processing
train_df['label_name'] = train_df['label'].apply(lambda x: label_names[x])
test_df['label_name'] = test_df['label'].apply(lambda x: label_names[x])

# Save to CSV for manual inspection
train_df.to_csv("data_train_raw.csv", index=False)
test_df.to_csv("data_test_raw.csv", index=False)

print("Saved data_train_raw.csv and data_test_raw.csv with columns: ", train_df.columns.tolist())
print("Sample:")
print(train_df[['text','label','label_name']].head())
