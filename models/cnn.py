import torch
import torch.nn as nn
import torch.optim as optim
import utils


class CrisisClassifier(nn.Module):
    def __init__(self, input_dim):
        """
        Constructor for CrisisClassifier.

        Parameters
        ----------
        input_dim : int
            The number of features in the input data.
        """
        super(CrisisClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input data, of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            The output of the forward pass, of shape (batch_size, 2).
        """
        return self.fc(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, test_df = utils.load_data()
    X_train, X_test = utils.preprocess_text(train_df, test_df)
    y_train, y_test = utils.encode_labels(train_df, test_df)

    # Convert numpy arrays to PyTorch tensors and move them to the correct device
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    # Initialize model
    model = CrisisClassifier(input_dim=5000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with early stopping
    num_epochs = 50
    best_loss = float("inf")
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0  # Reset patience
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X_test), axis=1)
        utils.evaluate_model(y_test, y_pred, 'CNN')
