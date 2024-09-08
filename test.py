import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def correlation_coefficient(y_true, y_pred):
    # y_true = y_true.flatten()
    # y_pred = y_pred.flatten()
    return pearsonr(y_pred, y_true)[0]

def plot_pred_actual(test_dataset, model):
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_targets = all_targets.reshape(-1)
    all_preds = all_preds.reshape(-1)
    dates = test_dataset.dates
    plt.figure(figsize=(100,30))
    plt.plot(dates, all_preds, label="predicted")
    plt.plot(dates, all_targets, label="actual")
    plt.legend()
    plt.show()
    print(correlation_coefficient(all_preds[7:], all_targets[:-7]))

def test(model, test_dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), plot=False, denormalise=False):
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_targets = all_targets.reshape(-1)
    all_preds = all_preds.reshape(-1)
    
    if denormalise:
        all_targets = all_targets * 1000
        all_preds = all_preds * 1000

    mae = mean_absolute_error(all_targets, all_preds)
    smape_value = smape(all_targets, all_preds)
    correlation = correlation_coefficient(all_targets, all_preds)

    if plot:
        print(f"test loss:{mae}, sMAPE:{smape_value}, r:{correlation}")
    
    if mae < plot:
        # start_date = datetime.strptime(test_dataset.start[0], "%Y-%m-%d")
        # end_date = datetime.strptime(test_dataset.end[0], "%Y-%m-%d")
        start_date = test_dataset.start
        end_date = test_dataset.end
        dates = pd.date_range(start=start_date+timedelta(
            days=test_dataset.delta+test_dataset.tau+test_dataset.gamma), end=end_date)

        # dates = pd.date_range(start=test_dataset.start+timedelta(
        #     days=test_dataset.delta+test_dataset.tau-1), end=test_dataset.end)
        plt.figure(figsize=(10, 3))
        plt.plot(dates, all_preds, label="predicted")
        plt.plot(dates, all_targets, label="actual")
        plt.legend()
        plt.show()
        # ensure the plot is shown as a PDF when run in Jupyter Notebook

    if plot:
        return mae, smape_value, correlation, all_preds, all_targets
    return mae, smape_value, correlation
