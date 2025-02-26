import cupy as cp
from cuml.svm import SVC as cumlSVC
import utils


def move_to_gpu(X_train, X_test, y_train, y_test):
    """
    Transfers input data arrays to GPU memory using CuPy.

    Parameters
    ----------
    X_train : array-like
        Training features to be transferred to GPU.
    X_test : array-like
        Test features to be transferred to GPU.
    y_train : array-like
        Training labels to be transferred to GPU.
    y_test : array-like
        Test labels to be transferred to GPU.

    Returns
    -------
    tuple
        A tuple containing the GPU arrays: (X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu).
    """
    return (
        cp.asarray(X_train, dtype=cp.float32),
        cp.asarray(X_test, dtype=cp.float32),
        cp.asarray(y_train, dtype=cp.int32),
        cp.asarray(y_test, dtype=cp.int32),
    )


def train_and_evaluate_svm(X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu):
    """
    Train and evaluate a Support Vector Machine (SVM) classifier on GPU.

    Parameters
    ----------
    X_train_gpu : cupy.ndarray
        Training features stored in GPU memory.
    X_test_gpu : cupy.ndarray
        Test features stored in GPU memory.
    y_train_gpu : cupy.ndarray
        Training labels stored in GPU memory.
    y_test_gpu : cupy.ndarray
        Test labels stored in GPU memory.

    Returns
    -------
    None
        Prints the confusion matrix and accuracy of the SVM classifier, and plots the confusion matrix.

    Notes
    -----
    This function uses cuML's SVM implementation to leverage GPU acceleration for training and prediction.
    The predictions are transferred back to CPU for evaluation.
    GPU memory is cleaned up after evaluation.
    """

    svm_gpu = cumlSVC(kernel="rbf")
    svm_gpu.fit(X_train_gpu, y_train_gpu)
    y_pred_gpu = svm_gpu.predict(X_test_gpu)

    # Convert predictions back to CPU
    y_pred_cpu = y_pred_gpu.get()
    y_test_cpu = y_test_gpu.get()

    utils.evaluate_model(y_test_cpu, y_pred_cpu)

    # Clean up GPU memory
    del svm_gpu
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    train_df, test_df = utils.load_data()
    X_train, X_test = utils.preprocess_text(train_df, test_df)
    y_train, y_test = utils.encode_labels(train_df, test_df)
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = move_to_gpu(
        X_train, X_test, y_train, y_test)

    train_and_evaluate_svm(X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu)
