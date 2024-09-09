import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
from src.utils import set_seeds
from src.test import test
from torch.utils.data import DataLoader

def train(model, train_loader, learning_rate, val_dataset=None, test_dataset=None, warmup=10, patience=5, max_epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), denormalise=False, seed=42, plot=False, silent=True):
    set_seeds(seed)
    model.to(device)  # Move model to the specified device
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None
    train_log = []
    if test_dataset is not None:
        best_test_mae = float('inf')
    best_epoch = 0

    all_preds = []
    all_mae = []
    all_smape = []
    all_correlation = []
    all_train_loss = []
    all_val_loss = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # Move data and target to the specified device
            optimizer.zero_grad()  # Clear gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target.unsqueeze(1))  # Compute loss (target.unsqueeze(1) to match output shape)
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        if denormalise:
            train_loss *= 1000
        
        if test_dataset is not None:
            mae, smape_value, correlation, preds, targets = test(model, test_dataset, device, plot=best_test_mae, denormalise=denormalise)
            all_preds.append(preds)
            all_mae.append(mae)
            all_smape.append(smape_value)
            all_correlation.append(correlation)
        
        if val_dataset is not None:
            # validation phase
            val_mae, val_smape_value, val_correlation = test(model, val_dataset, denormalise=denormalise)
            # plt.figure(figsize=(10,3))
            # plt.plot(y_pred)
            # plt.plot(y_actual)
            # plt.show()
            # val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
            # model.eval()
            
            # all_preds = []
            # all_targets = []
            # with torch.no_grad():
            
            #     for data, target in val_loader:
            #         data, target = data.to(device), target.to(device)
            #         output = model(data)
            #         all_preds.append(output.cpu().numpy())
            #         all_targets.append(target.cpu().numpy())
            # val_loss = criterion(output, target.unsqueeze(1)) * 1000
            val_loss = val_mae

            log = f'Epoch {epoch}, Train Loss: {train_loss}, Validation MAE: {val_mae}, Validation SMAPE: {val_smape_value}, Validation Correlation: {val_correlation}'
            if not silent:
                print(log)
            train_log.append(log)
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            if plot and epoch % 5 == 0:
                plt.plot(all_train_loss, label="train_loss")
                plt.plot(all_val_loss, label="validation_loss")
                plt.plot(all_mae, label="test_loss")
                plt.legend()
                plt.show()

            # Check for early stopping
            if epoch >= warmup:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_train_loss = train_loss
                    epochs_no_improve = 0
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        else:
            log = f'Epoch {epoch}, Train Loss: {train_loss}'
            if not silent:
                print(log)
            train_log.append(log)
    plt.plot(all_train_loss, label="train_loss")
    plt.plot(all_val_loss, label="validation_loss")
    plt.plot(all_mae, label="test_loss")
    plt.legend()
    plt.show()
    # load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    if test_dataset is not None and val_dataset is None:
        return np.array(all_preds), np.array(all_mae), np.array(all_smape), np.array(all_correlation), np.array(all_train_loss)
    elif test_dataset is not None and val_dataset is not None:
        return best_epoch, np.array(all_preds), np.array(all_mae), np.array(all_smape), np.array(all_correlation), np.array(all_train_loss), np.array(all_val_loss)     

    print(f"Training complete with training loss: {best_train_loss}, validation loss: {best_loss}.")
    return train_log, best_loss, best_train_loss, best_epoch
