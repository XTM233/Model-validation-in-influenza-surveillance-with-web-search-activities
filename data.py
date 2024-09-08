import os
import copy
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import sample_blocks, label_weeks
from src.data_new import SubDataset

# set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_interpolate(df_ILI):
    """
    input: df_ILI is a dataframe, with columns "WeekStart", "WeekEnd", "ILIRate"
    """

    df_ILI['MidWeek'] = df_ILI['WeekStart'] + (df_ILI['WeekEnd'] - df_ILI['WeekStart']) / 2

    df_daily_ili = pd.DataFrame(columns=['Date', 'ILIRate'])

    # NOTE avoid loops, ie change to two Thursday -> 6 values in between
    for i in range(len(df_ILI) - 1):
        # define the start and end dates for interpolation
        start_date = df_ILI.iloc[i]['MidWeek']
        end_date = df_ILI.iloc[i + 1]['MidWeek']
        
        # generate a date range from start to end date
        dates = pd.date_range(start=start_date, end=end_date, inclusive="left")
        
        # perform linear interpolation for ILI rates
        rate_start = df_ILI.iloc[i]['ILIRate']
        rate_end = df_ILI.iloc[i + 1]['ILIRate']
        interpolated_rates = np.linspace(rate_start, rate_end, len(dates), endpoint=False)
        
        df_daily_ili = pd.concat([df_daily_ili, pd.DataFrame({'Date': dates, 'ILIRate': interpolated_rates})], ignore_index=True)

    df_daily_ili['Date'] = pd.to_datetime(df_daily_ili['Date'])

    return df_daily_ili

# def smooth(df, n=14):
#     """
#     Use past n days(inclusive) search query data to smooth search frequency, removing sparsity
#     """
#     base_weights = np.array([1 / (n - i) for i in range(n)])
    
#     def weighted_mean(x):
#         weights = base_weights[:len(x)]
#         weights /= weights.sum()
#         return np.dot(x, weights)
    
#     result_df = df.copy()
#     for column in df.columns:
#         result_df[column] = df[column].rolling(window=n, min_periods=1).apply(weighted_mean, raw=True)

#     return result_df

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

def preprocess_England():
    """
    Preprocess all England data, save in csv
    - Q_freq_smooth_2008-2019.csv
    - ILI_rates_daily_2008-2019.csv
    - queries_similarity.csv
    """
    root_path = "raw_data\\England\\"
    target_path = "processed_data\\England\\"

    with open(root_path + "dates.txt", 'r') as file:
        dates = [line.strip() for line in file]

    with open(root_path + 'queries.txt', 'r') as file:
        queries = [line.strip() for line in file]

    if os.path.exists(target_path + "ILI_rates_daily.csv"):
        df_daily_ili = pd.read_csv(target_path + "ILI_rates_daily.csv")
        df_daily_ili['Date'] = pd.to_datetime(df_daily_ili['Date'])
    else:
        df_ILI = pd.read_csv(root_path + 'ILI_rates_RCGP.csv')
        df_ILI['WeekStart'] = pd.to_datetime(df_ILI['WeekStart'])
        df_ILI['WeekEnd'] = pd.to_datetime(df_ILI['WeekEnd'])

        df_daily_ili = linear_interpolate(df_ILI)
        df_daily_ili.to_csv(target_path + "ILI_rates_daily.csv", index=False)

    if os.path.exists(target_path + "queries_similarity.csv"):
        similarity_scores = pd.read_csv(target_path + "queries_similarity.csv")
        similarity_scores = similarity_scores.set_index("index")
        similarity_scores["similarity"] = pd.to_numeric(similarity_scores["similarity"])
    else:
        embedding_path = target_path + 'embeddings.pt'
        model = SentenceTransformer('all-distilroberta-v1').to(device)
        seed_embeddings = model.encode(['flu', 'fever', 'flu', 'flu medicine', 'gp', 'hospital'], device=device)
        
        if os.path.exists(embedding_path):
            embeddings_tensor = torch.load(embedding_path, map_location=device)
            embeddings = embeddings_tensor.cpu().numpy()
        else:
            embeddings = model.encode(queries, device=device)
            embeddings_tensor = torch.tensor(embeddings).to(device)
            torch.save(embeddings_tensor, embedding_path)

        cosine_scores = torch.zeros(len(embeddings)).to(device)
        for seed_embedding in seed_embeddings:
            seed_embedding = torch.tensor(seed_embedding).to(device)
            cosine_scores += util.cos_sim(seed_embedding, embeddings).sum(dim=0)
        
        cosine_scores = cosine_scores.cpu().tolist()
        index_list = list(range(len(queries)))

        similarity_scores = pd.DataFrame({'index': index_list, 'query': queries, 'similarity': cosine_scores})
        similarity_scores.to_csv(target_path + "queries_similarity.csv", index=False)


    if os.path.exists(target_path + "Q_freq_pivot_smooth.csv"):
        pivot_df_q_smooth = pd.read_csv(target_path + "Q_freq_pivot_smooth.csv")
        pivot_df_q_smooth = pivot_df_q_smooth.set_index("date")
        pivot_df_q_smooth.index = pd.to_datetime(pivot_df_q_smooth.index)
        pivot_df_q_smooth.columns = pd.to_numeric(pivot_df_q_smooth.columns)
    else:
        if os.path.exists(target_path + "Q_freq_pivot.csv"):
            pivot_df = pd.read_csv(target_path + "Q_freq_pivot.csv", index_col="date")
            pivot_df.index = pd.to_datetime(pivot_df.index)
            pivot_df.columns = pd.to_numeric(pivot_df.columns)
        else:
            df_Q_freq = pd.read_csv(root_path + "Q_freq_sparse.csv", names=['date_id', 'query_id', 'frequency'], dtype={0: int, 1: int})

            df_Q_freq['date_id'] = df_Q_freq['date_id'] - 1
            df_Q_freq['query_id'] = df_Q_freq['query_id'] - 1
            df_query = df_Q_freq.copy(deep=True)
            df_query['date'] = df_query['date_id'].apply(lambda x: dates[x])
            df_query['query'] = df_query['query_id'].apply(lambda x: queries[x])
            df_query['date'] = pd.to_datetime(df_query['date'])
            df_query = df_query.sort_values(by="date")
            pivot_df = df_query.pivot(index=['date'], columns='query_id', values='frequency').fillna(0)
            pivot_df.to_csv(target_path + "Q_freq_pivot.csv")

        start_date = '2008-09-01'
        end_date = '2019-08-31'

        filtered_df = pivot_df.loc[start_date:end_date]

        # find columns where more than 90% of the data is 0
        columns_to_drop = filtered_df.columns[(filtered_df == 0).mean() > 0.9]

        # drop these columns from the original dataframe
        pivot_df.drop(columns=columns_to_drop, inplace=True)

        # create a filtered list of top 1000 queries according to similarity
        top_similarity_indices = similarity_scores.sort_values(by='similarity', ascending=False).index.tolist()
        top_queries = [index for index in top_similarity_indices if index in pivot_df.columns][:1000]
        pivot_df = pivot_df[top_queries]

        pivot_df_q_smooth = smooth_new(pivot_df)
        pivot_df_q_smooth.to_csv(target_path + "Q_freq_pivot_smooth.csv", index=True, index_label="date")


    return df_daily_ili, pivot_df_q_smooth, similarity_scores

def correlation_analysis(df_ili, pivot_df):
    """
    Computation correlations of search query frequencies with ILI rate.
    Output: vector of correlation scores, corresponding to queries
    """
    correlation_scores = []
    query_idx = []
    nan_count = [] 
    for i in pivot_df.columns:
        try:
            query_series = pivot_df.loc[:, i]
            combined_df = pd.concat([query_series, df_ili], axis=1)
            correlation = combined_df.corr().iloc[0, 1]
            if pd.isna(correlation):
                nan_count.append(i)
            else:
                correlation_scores.append(correlation)
                query_idx.append(i)
        except KeyError:
            # if i does not exist in pivot table
            print(f"index {i} does not exist in columns")
    print(f"Correlation for query {nan_count} is NaN")
    return correlation_scores, query_idx


# def composite(similarity, correlation):
#     # Both inputs are torch tensors
#     similarity_norm = (similarity - similarity.min()) / (similarity.max() - similarity.min())
#     correlation = torch.nan_to_num(correlation)
#     correlation_norm = (correlation - correlation.min()) / (correlation.max() - correlation.min())
#     bool_int = (correlation > 1e-9).to(torch.int)
#     composite_scores = bool_int * (similarity_norm**2 + correlation_norm**2).flatten()
#     return composite_scores

# NOTE unfiy blockdataset and subdataset, test/val results on the same season differ

class BlockDataset(Dataset):
    """
    Convert pre-smoothed/interpolated dataframes into block datasets that would be merged later for training/testing.
    Start and end dates inclusive.
    """
    def __init__(self, forecast_horizon, window_size, num_queries, start, end, latency=7, X="processed_data/England/Q_freq_pivot_smooth.csv", y="processed_data/England/ILI_rates_daily.csv", query_idx=None, feature_scaler=None):
        # Convert start and end to lists if they are not already
        # if isinstance(start, str):
        #     start = [start]
        # if isinstance(end, str):
        #     end = [end]

        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        
        self.pivot_df = self._load_data(X)
        self.ili_rate_df = self._load_data(y, is_target=True)
        
        self.delta = latency
        self.gamma = forecast_horizon
        self.tau = window_size
        self.m = num_queries
        self.feature_scaler = feature_scaler
        
        if query_idx is not None:
            self.query_idx = query_idx
        else:
            self.query_idx = self.select_query()[:self.m]

        self.pivot_df = self.pivot_df[self.query_idx]

        # Normalization
        if self.feature_scaler is not None:
            if isinstance(self.feature_scaler, dict) and 'min' in self.feature_scaler and 'max' in self.feature_scaler:
                # Apply Min-Max scaling using provided min and max values
                min_val = feature_scaler['min']
                max_val = feature_scaler['max']
                zero_range_columns = (max_val - min_val == 0)
                if any(zero_range_columns):
                    print("When creating test dataset")
                    for col in zero_range_columns.index[zero_range_columns]:
                        print(f"Skipping normalization for column {col} as min = max = {min_val[col]}")
                    non_zero_range_columns = zero_range_columns.index[~zero_range_columns]
                    self.pivot_df[non_zero_range_columns] = (self.pivot_df[non_zero_range_columns] - min_val[non_zero_range_columns]) / (max_val[non_zero_range_columns] - min_val[non_zero_range_columns])
                else:
                    self.pivot_df = (self.pivot_df - min_val) / (max_val - min_val)
                self.feature_scaler = feature_scaler
            elif self.feature_scaler == "minmax":
                # Compute and apply Min-Max scaling
                self.feature_scaler = {"min": None, "max": None}
                features = self.pivot_df.loc[:, self.query_idx]
                self.feature_scaler['min'] = features.min()
                self.feature_scaler['max'] = features.max()
                self.pivot_df[self.query_idx] = (features - self.feature_scaler['min']) / (self.feature_scaler['max'] - self.feature_scaler['min'])
            # elif self.feature_scaler == "standardisation":
            #     # Compute and apply standardization (z-score normalization)
            #     self.feature_scaler = {"mean": None, "std": None}
            #     features = self.pivot_df.loc[:, self.query_idx]
            #     self.feature_scaler['mean'] = features.mean()
            #     self.feature_scaler['std'] = features.std()
            #     self.pivot_df.loc[:, self.query_idx] = (features - self.feature_scaler['mean']) / self.feature_scaler['std']
            # elif isinstance(self.feature_scaler, dict) and 'mean' in self.feature_scaler and 'std' in self.feature_scaler:
            #     # Apply standardization using provided mean and std values
            #     features = self.pivot_df.loc[:, self.query_idx]
            #     self.pivot_df.loc[:, self.query_idx] = (features - self.feature_scaler['mean']) / self.feature_scaler['std']
            else:
                raise ValueError("Invalid feature_scaler provided.")
            self.ili_rate_df["ILIRate"] = self.ili_rate_df["ILIRate"]/1000 # ensure between 0 to 1

        # self.block_lengths = self.calculate_block_lengths()
        self.block_lengths = [len(self.pivot_df)]

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
        
        df = df.loc[self.start:self.end,:]
        return df

    # def calculate_block_lengths(self):
    #     block_lengths = []
    #     for start_date, end_date in zip(self.start, self.end):
    #         start_date = datetime.strptime(start_date, "%Y-%m-%d")
    #         end_date = datetime.strptime(end_date, "%Y-%m-%d")
    #         block_length = len(self.ili_rate_df[start_date - timedelta(days=self.delta): end_date - timedelta(days=self.delta)])
    #         block_lengths.append(block_length)
    #     return block_lengths

    def select_query(self, plot=False):
        # only compute correlation scores for data in the last 4 years for recency
        recent_data_start = self.ili_rate_df.index.max() - pd.DateOffset(years=4)
        # print(recent_data_start)
        recent_ili_rate_df = self.ili_rate_df[recent_data_start:]
        recent_pivot_df = self.pivot_df[recent_data_start:]
        correlation_scores, query_idx_exist = correlation_analysis(recent_ili_rate_df, recent_pivot_df)
        
        # similarity_scores = pd.read_csv("processed_data/England/queries_similarity.csv")
        # similarity_scores = similarity_scores.set_index("index")
        # similarity_scores["similarity"] = pd.to_numeric(similarity_scores["similarity"])
        # cosine_scores = similarity_scores.loc[query_idx_exist, "similarity"]
        # cosine_scores = torch.tensor(cosine_scores.values).to(device)
        correlation_scores = torch.tensor(correlation_scores).to(device)
        # scores = composite(cosine_scores, correlation_scores)

        query_idx_sorted = [query_idx_exist[i] for i in torch.argsort(correlation_scores, descending=True)]
        self.pivot_df = self.pivot_df[query_idx_exist]

        if plot:
            return query_idx_sorted, correlation_scores
        return query_idx_sorted

    # def select_validation(self, val_scheme=None):
    #     if len(self.start) != 1:
    #         raise AssertionError("No further split!")
    #     else:
    #         if val_scheme == "stratified":
    #             label_dict = label_weeks(self.ili_rate_df)
    #             start, end = sample_blocks(label_dict, 365, self.start[0], self.end[0])
    #         else:
    #             raise ValueError("Unsupported validation scheme")

    #     return start, end

    def create_datasets(self, val_scheme):
        val_start, val_end = val_scheme

        date_lists = []
        for i in range(len(val_start)):
            some_dates = pd.date_range(start=pd.to_datetime(val_start[i]), end=pd.to_datetime(val_end[i]))
            date_lists.append(some_dates)

        val_dates = set()
        for date_list in date_lists:
            val_dates.update(date_list)

        # Generate all dates within the range of self.start and self.end
        all_dates = pd.date_range(start=self.start, end=self.end)

        # Compute the training dates by excluding validation dates
        train_dates = set(all_dates) - val_dates

        # Convert sets back to sorted lists
        train_dates = sorted(list(train_dates))
        val_dates = sorted(list(val_dates))

        # Use deep copy for all initialisation arguments
        training_dataset = SubDataset(train_dates, copy.deepcopy(self.query_idx), self.m, all_start=self.start, all_end=self.end, pivot_df=self.pivot_df, ili_daily=self.ili_rate_df, forecast_horizon=self.gamma, window_size=self.tau, latency=self.delta, feature_scaler="minmax")
        
        validation_dataset = SubDataset(val_dates, copy.deepcopy(self.query_idx), self.m, all_start=self.start, all_end=self.end, pivot_df=self.pivot_df, ili_daily=self.ili_rate_df, forecast_horizon=self.gamma, window_size=self.tau, latency=self.delta, feature_scaler=copy.deepcopy(training_dataset.feature_scaler))

        return training_dataset, validation_dataset

    # def denormalize(self, data):
    #     if self.feature_scaler is not None and 'min' in self.feature_scaler and 'max' in self.feature_scaler:
    #         return data * (self.feature_scaler['max'] - self.feature_scaler['min']) + self.feature_scaler['min']
    #     else:
    #         raise ValueError("Unsupported feature scaler for denormalization")

    def __len__(self):
        # return sum(self.block_lengths) - len(self.start) * (self.delta + self.tau + self.gamma)
        return len(self.pivot_df) - (self.delta + self.tau + self.gamma)

    def __getitem__(self, idx):
        for i, length in enumerate(self.block_lengths):
            if idx < length - (self.delta + self.tau + self.gamma):
                # block_start_date = datetime.strptime(self.start[i], "%Y-%m-%d")
                block_start_date = self.start
                adjusted_idx = idx + self.delta + self.tau
                now_date = block_start_date + timedelta(days=adjusted_idx)

                # feature_rate = self.ili_rate_df.loc[now_date - timedelta(days=self.tau + self.delta - 1): now_date - timedelta(days=self.delta), 'ILIRate']
                # feature_rate = torch.tensor(feature_rate.values, dtype=torch.float).unsqueeze(1).to(device)
                features_df = self.pivot_df.loc[now_date - timedelta(days=self.tau - 1): now_date, self.query_idx]

                # if features_df.empty:
                #     combined_features = feature_rate.flatten()
                # else:
                features = torch.tensor(features_df.values, dtype=torch.float).to(device).flatten()
                # combined_features = torch.cat((feature_rate, features), dim=1).flatten()

                label = torch.tensor(self.ili_rate_df.loc[now_date + timedelta(days=self.gamma), 'ILIRate'], dtype=torch.float).to(device)

                return features, label
            idx -= (length - (self.delta + self.tau + self.gamma))

        raise IndexError("Index out of range")

if __name__ == "__main__":
    preprocess_England()
