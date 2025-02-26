import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_train.tsv"),
                       sep="\t", header=None, names=["text", "class"])
test_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_test.tsv"),
                      sep="\t", header=None, names=["text", "class"])

# Feature extraction
vectorizer = TfidfVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(train_df["text"]).toarray()
X_test = vectorizer.transform(test_df["text"]).toarray()
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["class"])
y_test = label_encoder.transform(test_df["class"])

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Define Neural Network Model


class CrisisClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CrisisClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Assuming binary classification
        )

    def forward(self, x):
        return self.fc(x)


# Initialize model
model = CrisisClassifier(input_dim=2000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = torch.argmax(model(X_test), axis=1)
    accuracy = (y_pred == y_test).sum().item() / y_test.size(0)

print(f'Accuracy: {accuracy * 100:.2f}%')
