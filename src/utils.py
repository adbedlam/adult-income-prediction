from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
def calc_metrics(y_true, y_pred, name):
    print(f'Посчитанные метрики для {name}: ')
    print("Accuracy: ", round(accuracy_score(y_true, y_pred), 4))
    print("Recall: ", round(recall_score(y_true, y_pred), 4))
    print("Precision: ", round(precision_score(y_true, y_pred), 4))
    print("F1: ", round(f1_score(y_true, y_pred), 4))
    print("ROC/AUC: ", round(roc_auc_score(y_true, y_pred), 4))

