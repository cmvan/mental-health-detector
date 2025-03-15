import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pprint
import cupy as cp


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_data(nrows=None):
    """
    Load crisis train and test datasets from base_dir.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing the crisis_train.tsv and crisis_test.tsv files.

    Returns
    -------
    train_df : pandas.DataFrame
        Contains the text and class columns of the train dataset.
    test_df : pandas.DataFrame
        Contains the text and class columns of the test dataset.
    """
    train_df = pd.read_csv(os.path.join(
        BASE_DIR, "crisis_train.tsv"), sep="\t", header=None, names=["text", "class"], nrows=nrows)
    test_df = pd.read_csv(os.path.join(
        BASE_DIR, "crisis_test.tsv"), sep="\t", header=None, names=["text", "class"], nrows=nrows)
    return train_df, test_df


def preprocess_text(train_df, test_df, max_features=5000):
    """
    Convert text data to TF-IDF features.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Contains the text and class columns of the train dataset.
    test_df : pandas.DataFrame
        Contains the text and class columns of the test dataset.
    max_features : int, optional (default=5000)
        The number of features to keep. If None, the number of features is unlimited.

    Returns
    -------
    X_train : array-like of shape (n_samples, n_features)
        The TF-IDF features of the train dataset.
    X_test : array-like of shape (n_samples, n_features)
        The TF-IDF features of the test dataset.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df["text"]).toarray()
    X_test = vectorizer.transform(test_df["text"]).toarray()
    return X_train, X_test


def encode_labels(train_df, test_df):
    """
    Encode the class labels of the train and test datasets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Contains the text and class columns of the train dataset.
    test_df : pandas.DataFrame
        Contains the text and class columns of the test dataset.

    Returns
    -------
    y_train : array-like of shape (n_samples,)
        The encoded class labels of the train dataset.
    y_test : array-like of shape (n_samples,)
        The encoded class labels of the test dataset.
    """

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["class"])
    y_test = label_encoder.transform(test_df["class"])
    return y_train, y_test


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


def evaluate_model(y_test, y_pred, model):
    """
    Evaluate the performance of the model.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        The true class labels of the test dataset.
    y_pred : array-like of shape (n_samples,)
        The predicted class labels of the test dataset.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test dataset.
    """
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_dict = classification_report(
        y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("Evaluation Metrics:")
    print("--------------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity (Recall): {cm[1, 1] / (cm[1, 1] + cm[1, 0]):.4f}")
    print(f"Specificity: {cm[0, 0] / (cm[0, 0] + cm[0, 1]):.4f}")
    print(
        f"Positive Predictive Value (PPV): {cm[1, 1] / (cm[1, 1] + cm[0, 1]):.4f}")
    print(
        f"Negative Predictive Value (NPV): {cm[0, 0] / (cm[0, 0] + cm[1, 0]):.4f}")
    print()
    print("Classification Report:")
    print("------------------------")
    pprint.pprint(classification_report_dict)

    plot_confusion_matrix(cm, model)


def plot_confusion_matrix(cm, model: str):
    """
    Plot the confusion matrix of the test labels and predicted labels.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        The true class labels of the test dataset.
    y_pred : array-like of shape (n_samples,)
        The predicted class labels of the test dataset.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model} Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Non-Suicide", "Suicide"])
    plt.yticks(tick_marks, ["Non-Suicide", "Suicide"])

    # Adding annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    filename = "_".join(model.lower().split(' '))
    plt.savefig(f"{filename}_confusion_matrix.png")
