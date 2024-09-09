import pandas as pd
import os
import torch
import json
import random
import numpy as np
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta

from src.data_new import load_dataframe

# manage experiment log files


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def create_experiment_dir(root, experiment_name):
    experiment_dir = os.path.join(root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_hyperparameter_ranges(hyperparameter_ranges, experiment_dir):
    with open(os.path.join(experiment_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameter_ranges, f, indent=4)

def save_fixed_specs(fixed_specs, experiment_dir):
    with open(os.path.join(experiment_dir, 'fixed_specs.json'), 'w') as f:
        json.dump(fixed_specs, f, indent=4)

def save_logs(train_log, experiment_dir): # NOTE write log during training
    with open(os.path.join(experiment_dir, 'train_log.txt'), 'w') as f:
        for string in train_log:
            f.write(string + "\n")

def save_model(model, experiment_dir):
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model.pth'))

# select validation blocks

def label_weeks(df_daily_ili, n=63, delta=2, by_season=True, test_season =2015):
    """
    Given data frame of daily ili rate, return the start and end of each "onset", "outset", "peak" blocks, 
    consider n consecutive days as one block. If by_season is True, split the data by seasons.
    """

    if by_season:
        # split the data into seasons
        seasons = {}
        
        # determine the range of years in the data
        # start_year = df_daily_ili.index.year.min()
        start_year = 2008
        # NOTE make start, end consistent with sample blocks
        # end_year = df_daily_ili.index.year.max()
        end_year = test_season
        
        for year in range(start_year, end_year):
            season_start = f'{year}-09-01'
            season_end = f'{year+1}-08-31'
            
            season_data = df_daily_ili[season_start:season_end]
            
            if not season_data.empty:
                blocks, central_dates = label_weeks(season_data, n, delta, by_season=False)
                seasons[year] = {'blocks': blocks, 'central_dates': central_dates}
        
        return seasons
    else:
        m = n // 2

        mu = df_daily_ili['ILIRate'].mean()
        sigma = np.sqrt(((df_daily_ili['ILIRate'] - mu) ** 2).mean())
        
        blocks = {"onset": [], "outset": [], "peak": []}
        central_dates = {"onset": [], "outset": [], "peak": []}
        
        current_pos = 0
        while current_pos < len(df_daily_ili):
            onset_candidates = df_daily_ili.iloc[current_pos:]
            onset_dates = onset_candidates[onset_candidates['ILIRate'] > mu + delta * sigma].index
            
            if onset_dates.empty:
                break
            
            onset_date = onset_dates[0]
            peak_candidates = df_daily_ili.loc[onset_date:]
            peak_date = peak_candidates['ILIRate'].idxmax()
            outset_candidates = df_daily_ili.loc[peak_date:]
            outset_dates = outset_candidates[outset_candidates['ILIRate'] < mu + 2 * sigma].index
            
            if outset_dates.empty:
                break
            
            outset_date = outset_dates[0]
            
            onset_block_start = onset_date - pd.Timedelta(days=m)
            onset_block_end = onset_date + pd.Timedelta(days=m)
            peak_block_start = peak_date - pd.Timedelta(days=m)
            peak_block_end = peak_date + pd.Timedelta(days=m)
            outset_block_start = outset_date - pd.Timedelta(days=m)
            outset_block_end = outset_date + pd.Timedelta(days=m)
            
            blocks["onset"].append((onset_block_start.strftime('%Y-%m-%d'), onset_block_end.strftime('%Y-%m-%d')))
            blocks["peak"].append((peak_block_start.strftime('%Y-%m-%d'), peak_block_end.strftime('%Y-%m-%d')))
            blocks["outset"].append((outset_block_start.strftime('%Y-%m-%d'), outset_block_end.strftime('%Y-%m-%d')))
            
            central_dates["onset"].append(onset_date.strftime('%Y-%m-%d'))
            central_dates["peak"].append(peak_date.strftime('%Y-%m-%d'))
            central_dates["outset"].append(outset_date.strftime('%Y-%m-%d'))
            
            current_pos = df_daily_ili.index.get_loc(outset_date) + 1
        
        return blocks, central_dates
    
def sample_blocks(season_dict, n, overall_start, overall_end):
    overall_start = pd.to_datetime(overall_start)
    overall_end = pd.to_datetime(overall_end)

    sampled_blocks = []

    categories = ['onset', 'peak', 'outset']
    years = list(season_dict.keys())

    if len(years) < 3:
        raise ValueError("There should be at least 3 different years in the season_dict to sample from.")

    # Ensure we sample from 3 different years
    sampled_years = random.sample(years, 3)

    for category, year in zip(categories, sampled_years):
        data = season_dict[year]
        valid_blocks = [
            (pd.to_datetime(start), pd.to_datetime(end)) 
            for start, end in data['blocks'][category] 
            if pd.to_datetime(start) >= overall_start and pd.to_datetime(end) <= overall_end
        ]
        if valid_blocks:
            sampled_blocks.append(random.choice(valid_blocks))
        else:
            print(data['blocks'][category])
    # merge overlapping blocks
    sampled_blocks = sorted(sampled_blocks, key=lambda x: x[0])
    merged_blocks = []

    current_start, current_end = sampled_blocks[0]
    for start, end in sampled_blocks[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_blocks.append((current_start, current_end))
            current_start = start
            current_end = end

    merged_blocks.append((current_start, current_end))

    # Compute the number of days for the sampled blocks
    m = sum((end - start).days + 1 for start, end in merged_blocks)

    # Sample the remaining days
    remaining_days = n - m

    if remaining_days > 0:
        remaining_dates = pd.date_range(overall_start, overall_end).difference(
            pd.date_range(overall_start, overall_end, freq='D').intersection(
                pd.Series([pd.date_range(start, end) for start, end in merged_blocks]).explode()
            )
        )

        remaining_blocks = []
        
        while remaining_days > 0 and len(remaining_dates) > 62:
            block_length = min(remaining_days, 63)
            
            # Ensure block_length is valid in the remaining_dates
            valid_starts = remaining_dates[:-block_length]
            if not valid_starts.empty:
                start_date = valid_starts[random.randint(0, len(valid_starts)-1)]
                end_date = start_date + pd.Timedelta(days=block_length - 1)

                # Ensure non-overlapping and valid gap
                if not merged_blocks or (start_date - merged_blocks[-1][1]).days >= 63:
                    remaining_blocks.append((start_date, end_date))
                    remaining_days -= block_length

                    remaining_dates = remaining_dates.difference(pd.date_range(start_date, end_date))

        merged_blocks.extend(remaining_blocks)

    # Convert to list of date strings
    start_dates = [start.strftime('%Y-%m-%d') for start, end in merged_blocks]
    end_dates = [end.strftime('%Y-%m-%d') for start, end in merged_blocks]

    return start_dates, end_dates

def standardize_and_sum(df, overall_start=None, overall_end=None, window_size=63, step=7):
    if overall_start is not None:
        start_date = datetime.strptime(overall_start, "%Y-%m-%d")
        end_date = datetime.strptime(overall_end, "%Y-%m-%d")
        df = df[start_date:end_date]

    df_standardized = (df - df.min()) / (df.max() - df.min())
    sliding_sums = []
    dates = []
    
    for start in range(0, len(df_standardized) - window_size + 1, step):
        window_sum = df_standardized.iloc[start:start + window_size].sum()
        sliding_sums.append(window_sum)
        dates.append(df.index[start])
    
    result_df = pd.DataFrame(sliding_sums, index=dates)
    return result_df

def ks_algorithm(df, num_vectors=6, no_summer=False, query_idx=None):
    # TODO pass query idx (a list of df column labels which are all integers), taking selected features only, when query idx is an empty list, choose with only target
    # TODO not restricting num_vectors but total number of days
    selected_vectors = []
    selected_dates = []
    selected_indices = set()

    if no_summer:
        df = df[~df.index.to_series().between(df.index.to_series().apply(lambda x: x.replace(month=6, day=1)),
                                              df.index.to_series().apply(lambda x: x.replace(month=8, day=31)))]

    max_dist = 0
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dist = euclidean(df.iloc[i], df.iloc[j])
            if dist > max_dist:
                max_dist = dist
                pair = (i, j)
    
    selected_vectors.extend([df.iloc[pair[0]], df.iloc[pair[1]]])
    selected_dates.extend([df.index[pair[0]], df.index[pair[1]]])
    selected_indices.update(pair)

    while len(selected_indices) < num_vectors:
        max_dist = 0
        mean_vector = np.mean(selected_vectors, axis=0)
        next_vector_index = None
        for i in range(len(df)):
            if i in selected_indices:
                continue
            dist = euclidean(df.iloc[i], mean_vector)
            if dist > max_dist:
                max_dist = dist
                next_vector_index = i
        
        if next_vector_index is not None:
            selected_vectors.append(df.iloc[next_vector_index])
            selected_dates.append(df.index[next_vector_index])
            selected_indices.add(next_vector_index)

    start_dates = [date.strftime('%Y-%m-%d') for date in selected_dates]
    end_dates = [(date + pd.Timedelta(days=62)).strftime('%Y-%m-%d') for date in selected_dates]
    sorted_dates = sorted(zip(start_dates, end_dates), key=lambda x: pd.to_datetime(x[0]))
    start_dates, end_dates = zip(*sorted_dates)

    merged_start_dates = []
    merged_end_dates = []
    
    current_start = start_dates[0]
    current_end = end_dates[0]

    for i in range(1, len(start_dates)):
        if pd.to_datetime(start_dates[i]) <= pd.to_datetime(current_end):
            current_end = max(current_end, end_dates[i])
        else:
            merged_start_dates.append(current_start)
            merged_end_dates.append(current_end)
            current_start = start_dates[i]
            current_end = end_dates[i]

    merged_start_dates.append(current_start)
    merged_end_dates.append(current_end)

    return merged_start_dates, merged_end_dates

def get_val_range(validation_scheme, test_season, seed, all_start= "2008-09-01", feature_path=None):
    all_end = str(test_season) + "-08-31"
    if validation_scheme == "last_block":
        val_start = [str(test_season - 1) + "-09-01"]
        val_end = [all_end]
    elif validation_scheme == "stratified":
        df = pd.read_csv("processed_data/England/ILI_rates_daily.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date")
        label_dict = label_weeks(df, test_season=test_season)
        # set_seeds(seed)
        set_seeds(42)
        val_start, val_end = sample_blocks(label_dict, 365, "2008-09-01", str(test_season) + "-08-31")

    elif validation_scheme == "stratified_small":
        df = pd.read_csv("processed_data/England/ILI_rates_daily.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date")
        label_dict = label_weeks(df, test_season=test_season)
        # set_seeds(seed)
        set_seeds(42)
        val_start, val_end = sample_blocks(label_dict, 240, "2008-09-01", str(test_season) + "-08-31")

    elif validation_scheme == "ks_alg":
        df, _ = load_dataframe(feature_path)
        # df = pd.read_csv('processed_data/England/Q_freq_pivot_smooth.csv', index_col='date', parse_dates=True)
        transformed_df = standardize_and_sum(df, all_start, all_end)
        val_start, val_end = ks_algorithm(transformed_df)

    elif validation_scheme == "ks_no_summer":
        df, _ = load_dataframe(feature_path)
        # df = pd.read_csv('processed_data/England/Q_freq_pivot_smooth.csv', index_col='date', parse_dates=True)
        transformed_df = standardize_and_sum(df, all_start, all_end)
        val_start, val_end = ks_algorithm(transformed_df, no_summer=True)

    elif validation_scheme == "all_summer":
        val_start = []
        val_end = []
        for year in range(test_season-3, test_season + 1):
            val_start.append(str(year) + "-06-01")
            val_end.append(str(year)+"-08-31")
    elif validation_scheme == "three_blocks":
        val_start = [f'{test_season-3}-09-01', f'{test_season-2}-12-10',f'{test_season}-03-20']
        val_end = [f'{test_season-3}-12-09', f'{test_season-1}-03-19', f'{test_season}-08-31']

    return val_start, val_end