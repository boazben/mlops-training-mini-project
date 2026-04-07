import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import json

# 1. Load the data
print("Loading data...")
df = pd.read_csv('data/dataset.csv')
X = df[['feature1', 'feature2']]
y = df['label']

# 2. Train a simple model
print("Training model...")
model = LogisticRegression()
model.fit(X, y)

# 3. Save the trained model
print("Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 4. Save the metrics
print("Saving metrics...")
metrics = {'accuracy': model.score(X, y)}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Done! Check for model.pkl and metrics.json")