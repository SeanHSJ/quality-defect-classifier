"""
Data loading utilities for the UCI SECOM semiconductor manufacturing dataset.
"""
import pandas as pd


def load_secom(data_path: str, labels_path: str):
    """
    Load the SECOM feature data and labels.

    Parameters
    ----------
    data_path : path to secom.data
    labels_path : path to secom_labels.data

    Returns
    -------
    X : DataFrame of 590 sensor features
    y : Series of labels, remapped to 0 = pass, 1 = fail
    timestamps : Series of timestamps from the labels file
    """
    X = pd.read_csv(data_path, sep=r"\s+", header=None)
    X.columns = [f"sensor_{i}" for i in range(1, X.shape[1] + 1)]

    labels_raw = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["label", "timestamp"])
    # Original encoding: -1 = pass, 1 = fail. Remap to 0 = pass, 1 = fail for standard sklearn convention.
    y = labels_raw["label"].map({-1: 0, 1: 1})
    timestamps = labels_raw["timestamp"]

    return X, y, timestamps


if __name__ == "__main__":
    X, y, timestamps = load_secom("../data/secom.data", "../data/secom_labels.data")
    print(X.shape, y.shape)
    print("Failure rate:", y.mean())
    print(X.head())
