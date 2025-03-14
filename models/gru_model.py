import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import utils  # Ensure utils.py has the required data processing functions
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output 1 neuron for binary classification
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        x, _ = torch.max(gru_out, dim=1)  # GlobalMaxPooling
        x = self.dropout(x)
        return self.fc(x).squeeze(1)  # Flatten output


# Load and Preprocess Data
train_df, test_df = utils.load_data(1000)
X_train, X_test = utils.preprocess_text(train_df, test_df)
y_train, y_test = utils.encode_labels(train_df, test_df)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, and optimizer
model = GRUModel(vocab_size=5000, embed_dim=300, hidden_size=60).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        if outputs.dim() > 1:
            outputs = outputs.squeeze(1) 
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
        
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
# print(f"Test Accuracy: {accuracy:.4f}")

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Extract TN, FP, FN, TP from confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate additional metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV
recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # Sensitivity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0        # Negative Predictive Value

# Print Results
print("\n===== Evaluation Metrics =====")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision (PPV): {precision:.4f}")
print(f"Negative Predictive Value (NPV): {npv:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

