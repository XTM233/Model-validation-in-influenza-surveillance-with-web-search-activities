import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import itertools
from src.utils import *
from src.train import train
from src.test import test
from src.data import BlockDataset
from src.network import SimpleFNN

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(seed, test_season, validation_scheme, learning_rate, batch_size, dropout_prob, hidden_layers, window_size, num_queries, forecast_horizon, patience, hidden_size, experiment_dir=None):
    # Experiment directory, specific to validation scheme
    # root_dir = os.path.join('.', 'experiments', 'England', validation_scheme)
    target_path = os.path.join('.', 'processed_data', 'England')
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # feature_path_processed=os.path.join(".", "alternative_processed_data", "England", "Q_freq_pivot_smooth_2000_filter_new.csv")
    # feature_path_processed = os.path.join(target_path, "Q_freq_pivot_smooth_1000_filtered.csv")
    feature_path_processed = os.path.join("reference_processed_data/England", "Q_freq_pivot_smooth_1000_filter_new.csv")
    if experiment_dir is None:
        experiment_name = f"{validation_scheme}_test{test_season}_seed{seed}_" + experiment_time

        experiment_dir = create_experiment_dir(os.path.join("experiments", "England"), experiment_name)

    # Define the hyperparameter ranges you experimented with
    hyperparameter_ranges = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'dropout_prob': dropout_prob,
        'hidden_layers': hidden_layers,
        "window_size": window_size,
        "num_queries": num_queries,
        "hidden_size": hidden_size
    }

    # Check if hyperparameter ranges lead to only one combination
    single_combination = all(len(v) == 1 for v in hyperparameter_ranges.values())

    # Define fixed specifications
    fixed_specs = {
        "forecast_horizon": forecast_horizon,
        "test_season": test_season,
        'output_size': 1,
        'max_epochs': 200,
        'patience': patience,
        "seed": seed,
        "val_scheme": validation_scheme,
        "feature_path": feature_path_processed
    }

    with open(os.path.join(experiment_dir, 'result.csv'), mode='w') as file:
        file.write('learning_rate,batch_size,dropout_prob,hidden_layers,window_size,num_queries, hidden_size,train_loss,val_loss,best_epoch\n')

    save_hyperparameter_ranges(hyperparameter_ranges, experiment_dir)
    save_fixed_specs(fixed_specs, experiment_dir) 

    keys = hyperparameter_ranges.keys()
    values = hyperparameter_ranges.values()

    # Create the hyperparameter space using itertools.product
    hyperparameter_space = list(itertools.product(*values))

    best_val_loss = float('inf')
    best_model_wts = None
    best_train_log = ""
    best_hyperparameters = {}

    val_start, val_end = get_val_range(validation_scheme,test_season, seed, feature_path=feature_path_processed)
    all_start = "2008-09-01"
    all_end = str(fixed_specs["test_season"]) + "-08-31"
 
    # TODO generate sorted indx

    with open(os.path.join(experiment_dir, 'val_set.txt'), mode='w') as file:
        file.write("val_start, val_end\n")
        for i in range(len(val_start)):
            file.write(f"{val_start[i]}, {val_end[i]}\n")
    
    # Create datasets for each unique num_queries value
    datasets = {}
    for num_query in set(num_queries):
        datasets[num_query] = BlockDataset(forecast_horizon=fixed_specs["forecast_horizon"], window_size=window_size[0], X=feature_path_processed, num_queries=num_query, start=all_start, end=all_end)

    for params in hyperparameter_space:
        hyperparameters = dict(zip(keys, params))
        print(hyperparameters)

        window_size = hyperparameters["window_size"]
        num_queries = hyperparameters["num_queries"]
        input_size = (window_size) * (num_queries)
        set_seeds(seed)
        # NOTE choice of first layer neuron vs num_queries
        model = SimpleFNN(input_size, output_size=fixed_specs["output_size"], hidden_layers=hyperparameters["hidden_layers"], neurons_per_layer=hyperparameters["hidden_size"], dropout_prob=hyperparameters["dropout_prob"], m=num_queries).to(device)
        for layer in model.children():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
     
        all_dataset = datasets[num_queries]
        train_dataset, val_dataset = all_dataset.create_datasets((val_start, val_end))
        # NOTE this should be passed to datasets list
        all_dataset.feature_scaler = train_dataset.feature_scaler

        set_seeds(seed)
        train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)


        set_seeds(seed)
        train_log, val_loss, train_loss, epoch = train(model, train_loader, learning_rate=hyperparameters["learning_rate"], val_dataset=val_dataset, patience=fixed_specs["patience"], warmup=20, max_epochs=fixed_specs["max_epochs"], device=device, denormalise=True, silent=False)
        # train_log, val_loss, train_loss = train(model, train_loader, val_loader, learning_rate=hyperparameters["learning_rate"], patience=fixed_specs["patience"], max_epochs=fixed_specs["max_epochs"], device=device)

        if single_combination:
            print(f"Validation Loss: {val_loss}")
            test_start = str(fixed_specs["test_season"])+"-09-01"
            test_end = str(fixed_specs["test_season"]+1)+"-08-31"
            test_dataset = BlockDataset(forecast_horizon=fixed_specs["forecast_horizon"], window_size=window_size, X=feature_path_processed, num_queries=hyperparameters["num_queries"], start=test_start, end=test_end, query_idx=train_dataset.query_indx, feature_scaler=train_dataset.feature_scaler)
            mae, smape_value, correlation, y_pred, y_true = test(model, test_dataset, denormalise=True, plot=True)

            return val_loss, mae, smape_value, correlation, y_pred, y_true
        if val_loss < best_val_loss:
            best_train_log = train_log
            best_epoch = epoch
            best_train_loss = train_loss
            # best_model = model
            best_hyperparameters = hyperparameters
            best_val_loss = val_loss
            save_logs(best_train_log, experiment_dir)
            save_model(model, experiment_dir)
            with open(os.path.join(experiment_dir, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_hyperparameters, f, indent=4)        

        with open(os.path.join(experiment_dir, 'result.csv'), mode='a') as file:
            file.write(f"{hyperparameters['learning_rate']},{hyperparameters['batch_size']},{hyperparameters['dropout_prob']},{hyperparameters['hidden_layers']},{hyperparameters['window_size']},{hyperparameters['num_queries']},{hyperparameters['hidden_size']},{train_loss},{val_loss}, {epoch}\n")

    test_start = str(fixed_specs["test_season"]) + "-09-01"
    test_end = str(fixed_specs["test_season"] + 1) + "-08-31"
    best_hyperparameters_path = os.path.join(experiment_dir, 'best_hyperparameters.json')
    if os.path.exists(best_hyperparameters_path):
        with open(best_hyperparameters_path, 'r') as f:
            best_hyperparameters = json.load(f)

    window_size = best_hyperparameters["window_size"]
    num_queries = best_hyperparameters["num_queries"]
    input_size = window_size * num_queries

    # NOTE only create dataset once
    all_dataset = BlockDataset(forecast_horizon=fixed_specs["forecast_horizon"],window_size=window_size, X=feature_path_processed, num_queries=num_queries, start=all_start, end=all_end)

    train_dataset, _ = all_dataset.create_datasets((val_start, val_end))
    
    test_start = str(fixed_specs["test_season"])+"-09-01"
    test_end = str(fixed_specs["test_season"]+1)+"-08-31"
    test_dataset = BlockDataset(forecast_horizon=fixed_specs["forecast_horizon"], window_size=window_size, X=feature_path_processed, num_queries=best_hyperparameters["num_queries"], start=test_start, end=test_end, query_idx=train_dataset.query_indx, feature_scaler=train_dataset.feature_scaler)

    best_model = SimpleFNN(input_size, output_size=1, hidden_layers=best_hyperparameters["hidden_layers"], neurons_per_layer=best_hyperparameters['hidden_size'], dropout_prob=0, m=num_queries)
    best_model.load_state_dict(torch.load(os.path.join(experiment_dir, 'best_model.pth')))

    mae, smape_value, correlation, y_pred, y_true = test(best_model, test_dataset, denormalise=True, plot=True)

    # save y_pred and y_true to files
    np.save(os.path.join(experiment_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(experiment_dir, 'y_true.npy'), y_true)

    with open(os.path.join(experiment_dir, 'test_result.csv'), mode='w') as file:
        file.write(
            f"seed,test_season,val_scheme,mae,smape,bivariate_correlation,epoch,learning_rate,num_layers,num_queries,batch_size\n"
            f"{fixed_specs['seed']},{fixed_specs['test_season']},{fixed_specs['val_scheme']},{mae}, {smape_value},{correlation},{best_epoch},"
            f"{best_hyperparameters['learning_rate']},{best_hyperparameters['hidden_layers']},{best_hyperparameters['num_queries']},{best_hyperparameters['batch_size']}\n") 

    print(f"Test on the best model for validation scheme {validation_scheme} season {fixed_specs['test_season']}")
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'sMAPE: {smape_value}')
    print(f'Correlation Coefficient: {correlation}')
    print(best_hyperparameters)

# TODO feature_path as args
# TODO plot true VS actual graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training and testing with specified seed and test season.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test_season', type=int, required=True, help='Test season year')
    parser.add_argument('--forecast_horizon', type=int, default=0, help='Forecast horizon')
    parser.add_argument("--validation_scheme", type=str, required=True, help="Choose from last_block, stratified, ks_alg")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument('--learning_rate', nargs='+', type=float, default=[0.001, 0.00001], help='Learning rate range')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[28, 56], help='Batch size range')
    parser.add_argument('--dropout_prob', nargs='+', type=float, default=[0], help='Dropout probability range')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[2, 4], help='Hidden layers range')
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[128, 256, 512], help='number of neurons per layer')
    parser.add_argument('--window_size', nargs='+', type=int, default=[14], help='Window size range')
    parser.add_argument('--num_queries', nargs='+', type=int, default=[100, 200], help='Number of queries range')
    parser.add_argument("--patience", nargs="?", type=int, default=10)

    args = parser.parse_args()

    main(args.seed, args.test_season, args.validation_scheme, args.learning_rate, args.batch_size, args.dropout_prob, args.hidden_layers, args.window_size, args.num_queries, args.forecast_horizon, args.patience, args.hidden_size, args.root_dir)
