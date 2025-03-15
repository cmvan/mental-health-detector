import utils
import cupy as cp
import cudf
from cuml.linear_model import LogisticRegression
from cuml.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


if __name__ == "__main__":
    # Load data
    train_df, test_df = utils.load_data()
    X_train, X_test = utils.preprocess_text(train_df, test_df)
    y_train, y_test = utils.encode_labels(train_df, test_df)

    # Move to GPU
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = utils.move_to_gpu(
        X_train, X_test, y_train, y_test
    )

    # Standardize features (GPU-accelerated)
    scaler = StandardScaler()
    X_train_gpu = scaler.fit_transform(X_train_gpu)
    X_test_gpu = scaler.transform(X_test_gpu)

    # Train Logistic Reg
    model = LogisticRegression(max_iter=500, penalty="l2", random_state=42)
    model.fit(X_train_gpu, y_train_gpu)

    # Make predictions (GPU-based)
    y_pred_gpu = model.predict(X_test_gpu)

    # Convert predictions back to CPU
    y_pred = y_pred_gpu.get()
    y_test_cpu = y_test_gpu.get()

    # Evaluation (CPU-based)
    print("Accuracy:", accuracy_score(y_test_cpu, y_pred))
    print(classification_report(y_test_cpu, y_pred))

    # Use utils' evaluation function if needed
    utils.evaluate_model(y_test_cpu, y_pred)
