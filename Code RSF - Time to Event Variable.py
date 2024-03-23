%reset

import pandas as pd

def calculate_time_to_event(df):
    # Make a copy of the original DataFrame
    df_copy = df.copy()
    
    # Convert 'period' to datetime format if it's not already
    df_copy['period'] = pd.to_datetime(df_copy['period'])
    
    # Calculate max period for every unique_id
    max_period_by_id = df_copy.groupby('unique_id')['period'].max().reset_index()
    
    # For churners, take the minimum period of churn per unique_id
    churned_df = df_copy[df_copy['final_churn'] == 1]
    min_period_by_id = churned_df.groupby('unique_id')['period'].min().reset_index()
    
    # Merge max period for churners back into the original DataFrame
    max_period_by_id = max_period_by_id.merge(min_period_by_id, on='unique_id', how='left', suffixes=('_max', '_min'))
    max_period_by_id['max_period'] = max_period_by_id['period_min'].fillna(max_period_by_id['period_max'])
    
    # Remove periods after the first period of churn for churners
    churners_max_periods = max_period_by_id.set_index('unique_id')['max_period']
    df_copy.loc[df_copy['unique_id'].isin(churners_max_periods.index), 'max_period'] = df_copy['unique_id'].map(churners_max_periods)
    df_copy = df_copy[df_copy['period'] <= df_copy['max_period']]
    
    # Merge max period for all unique_ids back into the original DataFrame
    non_churners_max_periods = max_period_by_id.set_index('unique_id')['period_max']
    df_copy.loc[df_copy['unique_id'].isin(non_churners_max_periods.index), 'max_period'] = df_copy['unique_id'].map(non_churners_max_periods).fillna(df_copy['max_period'])
    
    # Set max_period to current period for rows where final_churn is equal to 1
    df_copy.loc[df_copy['final_churn'] == 1, 'max_period'] = df_copy.loc[df_copy['final_churn'] == 1, 'period']
    
    # Calculate time to event in months
    df_copy.loc[:, 'time_to_event'] = (df_copy['max_period'] - df_copy['period']).dt.days / 30.437
    
    # Drop max_period column if not needed anymore
    df_copy.drop(columns=['max_period'], inplace=True)
    
    return df_copy

df_rsf = calculate_time_to_event(df_rsf)
