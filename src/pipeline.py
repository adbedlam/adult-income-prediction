from inference import inference_res, inference_res_with_shap
from train import train_save_XG, train_save_ANN, train_save_RFC
import os

models_path = ['../models/RFC.pkl', '../models/model.pth', '../models/XGBoost.pkl']

flag = False

count = 0

for model in models_path:
    if os.path.isfile(model):
        count += 1

    if count == 3:
        flag_inp = input("Модели обучены! Хотите переобучить? 1/0: \n")

        flag = True if flag_inp=='1' else False

train_save_RFC(flag)
train_save_ANN(flag)
train_save_XG(flag)

data = input("Введите строку данных через ',': \n")
RFC, XG, ANN = inference_res_with_shap(data)

print(f"Random Forest classifier: {RFC[0]} \nXGBoost: {XG[0]},\nSimple ANN: {ANN[0]}")