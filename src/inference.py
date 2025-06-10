import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import shap
import torch

from datapreprocessing import inference_preproc

Path("../reports/figures/").mkdir(parents=True, exist_ok=True)

def load_model():
    with open('../models/RFC.pkl', 'rb') as f:
        modelRFC = pickle.load(f)

    with open('../models/XGBoost.pkl', 'rb') as f:
        modelXGB = pickle.load(f)

    model = torch.load("../models/model.pth", weights_only=False)
    return modelRFC, modelXGB, model


def plot_shap_summary(shap_values, features, feature_names, model_name):
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"SHAP {model_name}")
    plt.tight_layout()
    plt.savefig(f"../reports/figures/shap_{model_name.lower()}.png")
    plt.close()


def generate_shap_explanations(X, feature_names, modelRFC, modelXGB, modelANN):
    # RF
    explainer_rfc = shap.Explainer(modelRFC.predict_proba, X)
    shap_values_rfc = explainer_rfc(X)

    # XGBoost
    explainer_xgb = shap.TreeExplainer(modelXGB)
    shap_values_xgb = explainer_xgb.shap_values(X)

    # ANN
    explainer_ann = shap.DeepExplainer(modelANN, torch.tensor(X, dtype=torch.float32))
    shap_values_ann = explainer_ann.shap_values(torch.tensor(X, dtype=torch.float32))

    # Построение графиков
    plot_shap_summary(shap_values_rfc, X, feature_names, "Random Forest")
    plot_shap_summary(shap_values_xgb, X, feature_names, "XGBoost")
    plot_shap_summary(shap_values_ann, X, feature_names, "ANN")


def inference_res(data):
    X, names = inference_preproc(data)

    modelRFC, modelXGB, ANN = load_model()

    ANN.eval()

    with torch.no_grad():
        output = ANN(torch.tensor(X, dtype=torch.float32))

    return (modelRFC.predict(X), modelXGB.predict(X), output), X, names, modelRFC, modelXGB, ANN


def inference_res_with_shap(data):
    preds, X, names, modelRFC, modelXGB, ANN = inference_res(data)
    generate_shap_explanations(X, names, modelRFC, modelXGB, ANN)
    return preds[0], preds[1], preds[2]