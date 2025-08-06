import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List
from preprocessing import load_generators, get_categories, recycling_categories

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


def test_data():
    generators = load_generators("../data/processed")
    X_test, y_test = generators['test_data']

    return X_test, y_test

def get_labels(y_test):
    categories = get_categories()
    label_categories = {}

    for idx, category in enumerate(categories):
        label_categories[idx] = category
    true_categories = []

    for label in y_test:
        true_categories.append(label_categories[label])
    binary_true = []
    for category in true_categories:    
        binary_true.append(recycling_categories.get(category, False))

    return binary_true

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    result = {'accuracy': accuracy,'precision': precision,'recall': recall,'f1_score': f1}

    return result

def performance_results():
    X_test, y_test = test_data()
    y_true = get_labels(y_test)
    
    model_output = {'Naive': '../data/output/naive_output.csv', 'Traditional': '../data/output/traditional_output.csv', 'Deep Learning': '../data/output/deep_learning_output.csv'}
    results = {}
    
    for model_name, csv_path in model_output.items():
        df = pd.read_csv(csv_path)
        y_pred = df['prediction'].tolist()
        metrics = calculate_metrics(y_true, y_pred)
        results[model_name] = metrics
    
    return results

def evaluation_report(results, output_path="../data/output"):
    report_path = os.path.join(output_path, "model_evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write("\n")

def main():
    results = performance_results()
    
    if results:
        evaluation_report(results)

if __name__ == "__main__":
    main() 