# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:33:05 2026

@author: WKS
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train():
    df = pd.DataFrame(load_iris(as_frame=True).frame)
    df.to_csv("data/iris.csv", index=False)

    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc:.4f}")
    return model, acc

if __name__ == "__main__":
    train()