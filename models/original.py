import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import numpy as np


def csr_to_sparse_tensor(csr):
    coo = csr.tocoo()  # Convert to COO format
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    size = torch.Size(csr.shape)
    return torch.sparse.FloatTensor(indices, values, size)


class LassoNN(nn.Module):
    def __init__(self):
        super(LassoNN, self).__init__()
        self.fc1 = nn.Linear(5000, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class SparseDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X_sparse = X_sparse  # Keep as sparse matrix
        self.y = torch.tensor(y, dtype=torch.float32)  # Convert y to tensor

    def __len__(self):
        return self.X_sparse.shape[0]

    def __getitem__(self, idx):
        X_dense = torch.tensor(self.X_sparse[idx].toarray(
        ), dtype=torch.float32).squeeze(0)  # Convert one row to dense
        return X_dense, self.y[idx]


# Training function
def train_model(model, criterion, optimizer, train_loader, lambda_reg=1e-4, epochs=20):
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            # Apply L1 regularization
            l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
            loss += lambda_reg * l1_norm

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return epoch_losses


def cross_val_fold(lambda_reg, max_df, min_df, X, y, kf, fold_idx):
    print(
        f"Training fold {fold_idx} with lambda={lambda_reg}, max_df={max_df}, min_df={min_df}")
    train_idx, val_idx = list(kf.split(X))[fold_idx]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Vectorize text data (remains sparse)
    vectorizer = TfidfVectorizer(
        max_df=max_df, min_df=min_df, max_features=5000)
    X_train_dtm = vectorizer.fit_transform(X_train)  # Sparse matrix
    X_val_dtm = vectorizer.transform(X_val)  # Sparse matrix

    # Use SparseDataset instead of converting to dense
    train_dataset = SparseDataset(X_train_dtm, y_train)
    val_dataset = SparseDataset(X_val_dtm, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model training and evaluation
    model = LassoNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005)

    train_model(model, criterion, optimizer, train_loader,
                lambda_reg=lambda_reg, epochs=5)

    # Evaluate on validation set
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            outputs = model(inputs)
            y_pred.extend(torch.round(torch.sigmoid(outputs)).numpy())

    score = utils.score_model(y_val, y_pred)
    return score


# Use parallelization to speed up cross-validation
def cross_val(X, y, lambda_vals, max_df_vals, min_df_vals, k_folds=10):
    best_lambda = None
    best_max_df = None
    best_min_df = None
    best_score = float('inf')

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    param_combinations = [(lambda_reg, max_df, min_df)
                          for lambda_reg in lambda_vals for max_df in max_df_vals for min_df in min_df_vals]
    results = Parallel(n_jobs=-1)(
        delayed(cross_val_fold)(lambda_reg, max_df, min_df, X, y, kf, fold_idx)
        for fold_idx in range(kf.get_n_splits(X))
        for lambda_reg, max_df, min_df in param_combinations
    )
    print('Results made!')

    for (lambda_reg, max_df, min_df), res in zip(param_combinations, results):
        avg_score = np.mean(res)
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = lambda_reg
            best_max_df = max_df
            best_min_df = min_df

    return best_lambda, best_max_df, best_min_df


if __name__ == "__main__":
    # Load data
    train_df, test_df = utils.load_data()
    X_train, X_test = train_df['text'], test_df['text']
    y_train, y_test = utils.encode_labels(train_df, test_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cross-validation parameters
    lambda_vals = [1e-5, 1e-4, 1e-3]  # Regularization values
    max_df_vals = [0.7, 0.8, 0.9]  # max_df for sparsity
    min_df_vals = [1, 5, 10]  # min_df for sparsity

    # Perform cross-validation
    best_lambda, best_max_df, best_min_df = cross_val(
        X_train, y_train, lambda_vals, max_df_vals, min_df_vals)

    # Final training with the best hyperparameters found from cross-validation
    print(
        f"Best lambda: {best_lambda}, Best max_df: {best_max_df}, Best min_df: {best_min_df}")

    vectorizer = TfidfVectorizer(
        max_df=best_max_df, min_df=best_min_df, max_features=5000)
    X_train_dtm = vectorizer.fit_transform(X_train)  # Sparse
    X_test_dtm = vectorizer.transform(X_test)  # Sparse

    # Use SparseDataset instead of converting to dense
    train_dataset = SparseDataset(X_train_dtm, y_train)
    test_dataset = SparseDataset(X_test_dtm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train final model
    model = LassoNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005)

    train_model(model, criterion, optimizer, train_loader,
                lambda_reg=best_lambda, epochs=10)

    # Evaluate final model
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            y_pred.extend(torch.round(torch.sigmoid(outputs)).numpy())

    utils.evaluate_model(y_test, y_pred, "L1 Regularized Logistic Regression")
