"""
Sub : 704 : AI_ML Practical Exercise
Practical 2: Bayesian Network for Medical Diagnosis (Heart Disease Dataset)

Problem Statement:
Write a program to construct a Bayesian network considering medical data. 
Use this model to demonstrate the diagnosis of heart patients using standard 
Heart Disease Data Set. Calculate the accuracy, precision, and recall for your 
data set. (dataset: heart.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Compatibility check for both newer and older pgmpy versions
try:
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


def main():
    print("=" * 65)
    print("STEP 1: LOADING & EXPLORING DATASET (heart.csv)")
    print("=" * 65)
    
    df = pd.read_csv("heart.csv")
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Drop irrelevant index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    
    # Map target variable AHD (Atherosclerotic Heart Disease)
    df["AHD"] = df["AHD"].map({"No": 0, "Yes": 1})
    
    # Select key clinical diagnostic attributes
    features = ["Age", "Sex", "ChestPain", "MaxHR", "ExAng", "Oldpeak", "AHD"]
    data = df[features].copy()
    
    # Handle missing values
    data = data.dropna()
    
    # Encode categorical feature 'ChestPain'
    data["ChestPain"] = data["ChestPain"].astype("category").cat.codes
    
    # Discretize continuous features into 3 quantile bins (Low=0, Medium=1, High=2)
    for col in ["Age", "MaxHR", "Oldpeak"]:
        data[col] = pd.qcut(data[col], q=3, labels=False, duplicates="drop")
        
    print("\nPreprocessed Data Sample (First 5 records):")
    print(data.head())
    
    print("\n" + "=" * 65)
    print("STEP 2: SPLITTING INTO TRAINING & TESTING SETS (80% / 20%)")
    print("=" * 65)
    
    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["AHD"]
    )
    print(f"Training set: {train.shape[0]} samples")
    print(f"Testing set:  {test.shape[0]} samples")
    
    print("\n" + "=" * 65)
    print("STEP 3: CONSTRUCTING THE BAYESIAN NETWORK (DAG)")
    print("=" * 65)
    
    # Bayesian Network Structure (Directed Acyclic Graph)
    # The presence of Heart Disease (AHD) causally influences the clinical indicators
    edges = [
        ("AHD", "Age"),
        ("AHD", "Sex"),
        ("AHD", "ChestPain"),
        ("AHD", "MaxHR"),
        ("AHD", "ExAng"),
        ("AHD", "Oldpeak")
    ]
    model = DiscreteBayesianNetwork(edges)
    print("Model Nodes:", model.nodes())
    print("Model Edges:", model.edges())
    
    print("\n" + "=" * 65)
    print("STEP 4: PARAMETER ESTIMATION (LEARNING CPDS)")
    print("=" * 65)
    
    estimator = BayesianEstimator(model, train)
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
    model.add_cpds(*cpds)
    
    is_valid = model.check_model()
    print("Bayesian Network Model Check Valid? :", is_valid)
    print("\nLearned CPD for ChestPain given AHD:")
    print(model.get_cpds("ChestPain"))
    
    print("\n" + "=" * 65)
    print("STEP 5: DEMONSTRATING HEART PATIENT DIAGNOSIS")
    print("=" * 65)
    
    infer = VariableElimination(model)
    
    # Case 1: High-Risk Patient
    high_risk_patient = {
        "Age": 2,        # High age quantile
        "Sex": 1,        # Male
        "ChestPain": 0,  # Asymptomatic chest pain (highest risk)
        "MaxHR": 0,      # Low maximum heart rate
        "ExAng": 1,      # Exercise-induced angina present
        "Oldpeak": 2     # High ST depression
    }
    q_high = infer.query(variables=["AHD"], evidence=high_risk_patient, show_progress=False)
    print("\n[Case 1] High-Risk Patient Evidence:", high_risk_patient)
    print(f"P(No Heart Disease): {q_high.values[0]:.4f} ({q_high.values[0]*100:.2f}%)")
    print(f"P(Heart Disease)   : {q_high.values[1]:.4f} ({q_high.values[1]*100:.2f}%)")
    print(f"Diagnosis Outcome  : {'POSITIVE FOR HEART DISEASE' if q_high.values.argmax() == 1 else 'NEGATIVE'}")
    
    # Case 2: Low-Risk Patient
    low_risk_patient = {
        "Age": 0,        # Low age quantile
        "Sex": 0,        # Female
        "ChestPain": 1,  # Non-anginal chest pain
        "MaxHR": 2,      # High maximum heart rate
        "ExAng": 0,      # No exercise-induced angina
        "Oldpeak": 0     # Low ST depression
    }
    q_low = infer.query(variables=["AHD"], evidence=low_risk_patient, show_progress=False)
    print("\n[Case 2] Low-Risk Patient Evidence:", low_risk_patient)
    print(f"P(No Heart Disease): {q_low.values[0]:.4f} ({q_low.values[0]*100:.2f}%)")
    print(f"P(Heart Disease)   : {q_low.values[1]:.4f} ({q_low.values[1]*100:.2f}%)")
    print(f"Diagnosis Outcome  : {'POSITIVE FOR HEART DISEASE' if q_low.values.argmax() == 1 else 'NEGATIVE'}")
    
    # Case 3: Incomplete / Partial Clinical Evidence
    partial_patient = {
        "Age": 2,
        "ChestPain": 0
    }
    q_partial = infer.query(variables=["AHD"], evidence=partial_patient, show_progress=False)
    print("\n[Case 3] Patient with Incomplete Evidence (Age & Chest Pain only):")
    print(f"P(Heart Disease)   : {q_partial.values[1]:.4f} ({q_partial.values[1]*100:.2f}%)")
    
    print("\n" + "=" * 65)
    print("STEP 6: MODEL EVALUATION ON TEST DATASET")
    print("=" * 65)
    
    evidence_cols = ["Age", "Sex", "ChestPain", "MaxHR", "ExAng", "Oldpeak"]
    predictions = []
    
    for _, row in test.iterrows():
        evidence = {col: row[col] for col in evidence_cols}
        q = infer.query(variables=["AHD"], evidence=evidence, show_progress=False)
        predictions.append(int(q.values.argmax()))
        
    y_test = test["AHD"].astype(int).values
    predictions = np.array(predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["No Heart Disease (0)", "Heart Disease (1)"]))


if __name__ == "__main__":
    main()
