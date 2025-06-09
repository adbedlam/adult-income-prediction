import pickle
import torch
from datapreprocessing import inference_preproc

def load_model():
    with open('../models/RFC.pkl', 'rb') as f:
        modelRFC = pickle.load(f)

    with open('../models/XGBoost.pkl', 'rb') as f:
        modelXGB = pickle.load(f)


    model = torch.load("../models/model.pth", weights_only=False)
    return modelRFC, modelXGB, model


def inference_res(data):
    X, X_ann = inference_preproc(data)

    modelRFC, modelXGB, ANN = load_model()

    ANN.eval()

    with torch.no_grad():
        output = ANN(torch.tensor(X, dtype=torch.float32))

    return modelRFC.predict(X), modelXGB.predict(X), output

