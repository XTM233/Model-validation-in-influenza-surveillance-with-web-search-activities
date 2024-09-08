# import os
from datetime import datetime, timedelta
# from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from src.utils import sample_blocks, label_weeks
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dates(start, end):
    if isinstance(start, str):
        start = [start]
    if isinstance(end, str):
        end = [end]
    
    all_dates = []
    for start_date, end_date in zip(start, end):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        all_dates.append(date_range)
    
    return all_dates

def inverse_dates(start, end, overall_start, overall_end):
    result_start = [overall_start] + [date + timedelta(days=1) for date in end]
    result_end = [date - timedelta(days=1) for date in start] + [overall_end]
    return result_start, result_end

def correlation_analysis(df_ili, pivot_df, plot=False, include_nan=False):
    """
    Computation correlations of search query frequencies with ILI rate.
    Output: vector of correlation scores, corresponding to queries
    """
    correlation_scores = []
    query_idx = []   
    for i in pivot_df.columns:
        try:
            query_series = pivot_df.loc[:, i]
            combined_df = pd.concat([query_series, df_ili], axis=1)
            correlation = combined_df.corr().iloc[0, 1]
            if pd.isna(correlation) and include_nan == False:
                if plot:
                    print(f"Correlation for index {i} is NaN")
                continue
            else:
                correlation_scores.append(correlation)
                query_idx.append(i)
        except KeyError:
            # if i does not exist in pivot table
                print(f"index {i} does not exist in columns")

    return correlation_scores, query_idx

def select_query(pivot_df, ili_daily, end_date, start_date="2008-09-01", return_score=False):
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    specified_start_date = pd.to_datetime(start_date)
    recent_data_start = max(end_date - pd.DateOffset(years=5), specified_start_date)

    recent_ili_rate_df = ili_daily[recent_data_start:end_date]
    recent_pivot_df = pivot_df[recent_data_start:end_date]

    if return_score:
        correlation_scores, query_idx = correlation_analysis(recent_ili_rate_df, recent_pivot_df, include_nan=True)
        # combine these two lists into one dataframe, columns "index", f"correlation_score_{end_date}" respectively
        result_df = pd.DataFrame({
            "index": query_idx,
            f"correlation_score_{end_date.strftime('%Y')}": correlation_scores
        })
        return result_df
    else:
        correlation_scores, query_idx_exist = correlation_analysis(recent_ili_rate_df, recent_pivot_df)
    
    correlation_scores = torch.tensor(correlation_scores).to(device)
    query_idx_sorted = [query_idx_exist[i] for i in torch.argsort(correlation_scores, descending=True)]

    return query_idx_sorted



class NewBlockDataset(Dataset):
    def __init__(self, dates, query_indx, num_query,pivot_df="processed_data/England/Q_freq_pivot_smooth.csv", ili_daily="processed_data/England/ILI_rates_daily.csv", forecast_horizon=0, window_size=14, latency=7, feature_scaler=None):
        if num_query <= len(query_indx):
            self.query_indx = query_indx[:num_query]
        elif num_query > len(query_indx):
            print("No enough queries")

        pivot_df = self._load_data(pivot_df)
        ili_daily = self._load_data(ili_daily, is_target=True)
        
        self.dates = [pd.to_datetime(date_list) for date_list in dates]
        flattened_dates = [index for sublist in self.dates for index in sublist]

        self.pivot_df = pivot_df.loc[flattened_dates, self.query_indx]
        self.ili_daily = ili_daily.loc[flattened_dates]

        self._normalize_data(feature_scaler)

        self.delta = latency
        self.gamma = forecast_horizon
        self.tau = window_size
        self.m = num_query
        self.start = dates[0][0]
        self.end = dates[-1][-1]

    def _load_data(self, data_path, is_target=False):

        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
            if is_target:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index("Date")
            else:
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
                df.columns = pd.to_numeric(df.columns)
        else:
            df = copy.deepcopy(data_path)
        return df

    def _normalize_data(self, feature_scaler):
        if max(self.ili_daily.max()) > 1:
            self.ili_daily /= 1000

        if feature_scaler is not None:
            if isinstance(feature_scaler, dict) and 'min' in feature_scaler and 'max' in feature_scaler:
                self.pivot_df = (self.pivot_df - feature_scaler['min']) / (feature_scaler['max'] - feature_scaler['min'])
                self.feature_scaler = feature_scaler
            elif feature_scaler == "minmax":
                self.feature_scaler = {"min": self.pivot_df.min(), "max": self.pivot_df.max()}
                self.pivot_df = (self.pivot_df - self.feature_scaler['min']) / (self.feature_scaler['max'] - self.feature_scaler['min'])

    def __len__(self):
        block_lengths = [len(d) for d in self.dates]
        return sum(block_lengths) - len(self.dates) * (self.delta + self.tau + self.gamma - 1)

    def __getitem__(self, idx):
        # Implement an explicit map from idx to date in self.dates (excluding the first self.delta+self.tau+self.gamma-1 dates in each nested list)
        for date_list in self.dates:
            block_length = len(date_list)
            if idx < block_length - (self.delta + self.tau + self.gamma - 1):
                adjusted_idx = idx + self.delta + self.tau - 1
                # print(adjusted_idx)
                now_date = date_list[adjusted_idx] # target is now_date + gamma
                # print(now_date)
                feature_rate = self.ili_daily.loc[now_date - timedelta(days=self.tau + self.delta - 1): now_date - timedelta(days=self.delta), 'ILIRate']
                feature_rate = torch.tensor(feature_rate.values, dtype=torch.float).unsqueeze(1).to(device)
                features_df = self.pivot_df.loc[now_date - timedelta(days=self.tau - 1): now_date, :]

                if features_df.empty:
                    combined_features = feature_rate.flatten()
                else:
                    features = torch.tensor(features_df.values, dtype=torch.float).to(device)
                    combined_features = torch.cat((feature_rate, features), dim=1).flatten()

                label = torch.tensor(self.ili_daily.loc[now_date + timedelta(days=self.gamma), 'ILIRate'], dtype=torch.float).to(device)

                return combined_features, label
            idx -= (block_length - (self.delta + self.tau + self.gamma - 1))

        raise IndexError("Index out of range")

class SubDataset(Dataset):
    def __init__(self, dates, query_indx, num_query, all_start="2008-09-01", all_end="2015-08-31", pivot_df="processed_data/England/Q_freq_pivot_smooth.csv", ili_daily="processed_data/England/ILI_rates_daily.csv", forecast_horizon=0, window_size=14, latency=7, feature_scaler=None):
        if num_query <= len(query_indx):
            self.query_indx = query_indx[:num_query]
        elif num_query > len(query_indx):
            print("No enough queries")

        pivot_df = self._load_data(pivot_df)
        ili_daily = self._load_data(ili_daily, is_target=True)

        self.delta = latency
        self.gamma = forecast_horizon
        self.tau = window_size
        self.m = num_query
        self.start = pd.to_datetime(all_start)
        self.end = pd.to_datetime(all_end)

        self.pivot_df = pivot_df.loc[self.start:self.end, self.query_indx]
        self.ili_daily = ili_daily.loc[self.start:self.end]

        self._normalize_data(feature_scaler)

        self.dates = [pd.to_datetime(date_list) for date_list in dates]
        self.valid_indices = self._compute_valid_indices()

    def _load_data(self, data_path, is_target=False):
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
            if is_target:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index("Date")
            else:
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
                df.columns = pd.to_numeric(df.columns)
        else:
            df = copy.deepcopy(data_path)
        return df

    def _normalize_data(self, feature_scaler):
        if max(self.ili_daily.max()) > 1:
            self.ili_daily /= 1000

        if feature_scaler is not None:
            # TODO unify feature scaler for both dataset class
            if isinstance(feature_scaler, dict) and 'min' in feature_scaler and 'max' in feature_scaler:
                min_val = feature_scaler['min']
                max_val = feature_scaler['max']
                zero_range_columns = (max_val - min_val == 0)
                if any(zero_range_columns):
                    print("When creating validation dataset")
                    for col in zero_range_columns.index[zero_range_columns]:
                        print(f"Skipping normalization for column {col} as min = max = {min_val[col]}")
                    non_zero_range_columns = zero_range_columns.index[~zero_range_columns]
                    self.pivot_df[non_zero_range_columns] = (self.pivot_df[non_zero_range_columns] - min_val[non_zero_range_columns]) / (max_val[non_zero_range_columns] - min_val[non_zero_range_columns])
                else:
                    self.pivot_df = (self.pivot_df - min_val) / (max_val - min_val)
                self.feature_scaler = feature_scaler
            elif feature_scaler == "minmax":
                self.feature_scaler = {"min": self.pivot_df.min(), "max": self.pivot_df.max()}
                min_val = self.feature_scaler['min']
                max_val = self.feature_scaler['max']
                zero_range_columns = (max_val - min_val == 0)
                if any(zero_range_columns):
                    print("When creating train dataset")
                    for col in zero_range_columns.index[zero_range_columns]:
                        print(f"Skipping normalization for column {col} as min = max = {min_val[col]}")
                    non_zero_range_columns = zero_range_columns.index[~zero_range_columns]
                    self.pivot_df[non_zero_range_columns] = (self.pivot_df[non_zero_range_columns] - min_val[non_zero_range_columns]) / (max_val[non_zero_range_columns] - min_val[non_zero_range_columns])
                else:
                    self.pivot_df = (self.pivot_df - min_val) / (max_val - min_val)


    def _compute_valid_indices(self):
        valid_indices = []

        for idx, target_date in enumerate(self.dates):
            feature_start_date = target_date - timedelta(days=self.tau + self.delta - 1)
            feature_end_date = target_date - timedelta(days=self.delta)
            if self.start <= feature_start_date and feature_end_date <= self.end:
                valid_indices.append((target_date, idx))
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_date = self.valid_indices[idx][0]
        feature_start_date = target_date - timedelta(days=self.tau + self.delta - 1)
        feature_end_date = target_date - timedelta(days=self.delta)

        features_df = self.pivot_df.loc[feature_start_date:feature_end_date, :]

        features = torch.tensor(features_df.values, dtype=torch.float).to(device).flatten()
        assert not torch.isnan(features).any(), "Features contain NaNs"

        label = torch.tensor(self.ili_daily.loc[target_date + timedelta(days=self.gamma), 'ILIRate'], dtype=torch.float).to(device)
        assert not torch.isnan(label).any(), "Label contains NaNs"

        return features, label


def load_dataframe(feature_path="processed_data/England/Q_freq_pivot_smooth.csv", target_path="processed_data/England/ILI_rates_daily.csv"):
    pivot_df = pd.read_csv(feature_path)
    df_daily_ili = pd.read_csv(target_path)
    pivot_df = pivot_df.set_index("date")
    pivot_df.columns = pd.to_numeric(pivot_df.columns)
    pivot_df.index = pd.to_datetime(pivot_df.index)

    df_daily_ili = pd.read_csv("processed_data/England/ILI_rates_daily.csv")
    df_daily_ili = df_daily_ili.set_index("Date")
    df_daily_ili.index = pd.to_datetime(df_daily_ili.index)
    
    return pivot_df, df_daily_ili

def smooth_new(df, n=14):
    """
    Use past n days (inclusive) search query data to smooth search frequency, removing sparsity.
    The weights are normalized weights of (1/13, ..., 1/3, 1/2, 1).
    """
    print("start")
    base_weights = np.array([1 / (n - i) for i in range(n)])
    
    def weighted_mean(window):
        weights = base_weights[-len(window):].copy()
        weights /= weights.sum()
        return np.dot(window, weights)
    
    result_df = df.copy()
    
    total = len(df.columns)
    count = 0
    for column in df.columns:
        smoothed_values = []
        for i in range(len(df)):
            window = df[column].iloc[max(0, i - n + 1):i + 1].values
            smoothed_values.append(weighted_mean(window))

        if count % 10 == 0:
            print(count/total)
        count += 1
        result_df[column] = smoothed_values

    return result_df

if __name__ == "__main__":
    # first load df
    pivot_df = pd.read_csv("processed_data/England/Q_freq_pivot_smooth.csv")
    pivot_df = pivot_df.set_index("date")
    pivot_df.columns = pd.to_numeric(pivot_df.columns)
    pivot_df.index = pd.to_datetime(pivot_df.index)

    df_daily_ili = pd.read_csv("processed_data/England/ILI_rates_daily.csv")
    df_daily_ili = df_daily_ili.set_index("Date")
    df_daily_ili.index = pd.to_datetime(df_daily_ili.index)
    query_idx_sorted = select_query(pivot_df, df_daily_ili, "2016-08-31")
    # get train_dates from val_dates
    train_dates = get_dates("2008-09-01", "2015-08-31")
    test_dates = get_dates("2015-09-01", "2016-08-31")
    train_dataset = NewBlockDataset(train_dates, query_idx_sorted, num_query=100, feature_scaler="minmax")
    # val_dataset = NewBlockDataset(val_dates, query_idx_sorted, feature_scaler=train_dataset.feature_scaler)
    test_dataset = NewBlockDataset(test_dates, query_idx_sorted, num_query=100, feature_scaler=train_dataset.feature_scaler)