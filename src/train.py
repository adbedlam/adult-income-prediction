import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xgboost import XGBClassifier

import os

from datapreprocessing import read_n_preproc
from utils import calc_metrics


class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



class ANet(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_dim):
        super(ANet, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(input_dim, 53),
            # nn.BatchNorm1d(87),
            nn.ReLU(),

            nn.Linear(53, 15),
            nn.ReLU(),

            nn.Linear(15, 18),
            nn.ReLU(),

            nn.Linear(18, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_test_model(model, train_loader, test_loader, lr, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred_out = model(x_batch)
            loss = criterion(y_pred_out.squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader)}")

    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred_out = model(x_batch).squeeze()
            preds.extend((y_pred_out > 0.5).int().cpu().tolist())
            targets.extend(y_batch.cpu().tolist())

    calc_metrics(targets, preds, "ANet")
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, )

    return acc


def train_save_XG(flag):
    model_path = '../models/XGBoost.pkl'
    if not os.path.isfile(model_path) or flag:
        print("Start train XGBoost")
        X_train, X_test, y_train, y_test = read_n_preproc()

        model_XGBoost = XGBClassifier(eval_metric='logloss', random_state=42)

        model_XGBoost.fit(X_train, y_train)

        y_pred = model_XGBoost.predict(X_test)

        calc_metrics(y_test, y_pred, "XGBoost")

        with open('../models/XGBoost.pkl', 'wb') as f:
            pickle.dump(model_XGBoost, f)

        print("End train XGBoost")
        print("Model XGBoost is saved")


def train_save_ANN(flag):
    model_path = '../models/model.pth'
    if not os.path.isfile(model_path) or flag:
        X_train, X_test, y_train, y_test = read_n_preproc()
        train_dataset = DataSet(X_train, y_train)
        test_dataset = DataSet(X_test, np.array(y_test))

        train_loader = DataLoader(train_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        model = ANet(input_dim=X_train.shape[1])
        train_test_model(model, train_loader, test_loader, 0.000032, 10)

        torch.save(model, '../models/model.pth')


def train_save_RFC(flag):
    model_path = '../models/RFC.pkl'
    if not os.path.isfile(model_path) or flag:
        X_train, X_test, y_train, y_test = read_n_preproc()

        RFC = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

        RFC.fit(X_train, y_train)

        RFC_pred = RFC.predict(X_test)

        calc_metrics(y_test, RFC_pred, "RFC")

        with open('../models/RFC.pkl', 'wb') as f:
            pickle.dump(RFC, f)
