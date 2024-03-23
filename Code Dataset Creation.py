######################################################### CODE PART 1 #########################################################


################################### PHASE 1 ###################################


###### PHASE 1: GENERAL ######

%reset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from scipy import stats
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox

# Loan csv file
path = '/Users/julianjager/Library/CloudStorage/GoogleDrive-julianjager22@gmail.com/My Drive/Master/Blok 3/Seminar/Data/Coolblue/churn_dataset_0109.csv'
df = pd.read_csv(path)


################################### PHASE 2 ###################################


###### PHASE 2.0: DROP COLUMNS ######

# Drop unnecessary columns
df = df.drop(['label_key', 
              'agreementtype', 
              'agreementidentifiersequence', 
              'main_product_bought', 
              'period_startdate', 
              'period_enddate', 
              'customer_segment'], 
             axis = 1)

###### PHASE 2.1: FEATURES - NUMERIC IDS ######

# Define numeric ids
df['id_debtor'] = pd.factorize(df['uuid_debtornumber'])[0]
df['id_contract'] = pd.factorize(df['uuid_contractidentifier'])[0]
df['id_ean'] = pd.factorize(df['uuid_ean'])[0]

###### PHASE 2.2: FEATURES - PERIODS ######

# Define delivery month and yeas as integers
df['delivery_month'] = df['delivery_month'].astype(int)
df['delivery_year'] = df['delivery_year'].astype(int)

# Define a period column
df['period'] = df['delivery_year'].astype(str) + '-' + df['delivery_month'].astype(str)
df['period'] = pd.to_datetime(df['period'], format='%Y-%m')
df = df.reset_index(drop=True)


# ###### PHASE 2.3: OPTIMIZE MEMORY ######

# Optimize memory
def optimize_memory(df):
    for old, new in [("integer", "unsigned"), ("float", "float")]:
        for col in df.select_dtypes(include=old).columns:
            df[col] = pd.to_numeric(df[col], downcast=new)

    return df

df = optimize_memory(df)

###### PHASE 2.4: FEATURES - UNIQUE IDENTIFYER ######

# Define a function that makes id's per group
def f_id_per_group(df):
    # Group by the specified columns
    grouped = df.groupby(['id_debtor', 'id_contract', 'commodity'])
    
    # Create the new DataFrame of the groups and the min and max period
    df_grouped = grouped.agg({
        'id_debtor': 'first',  
        'id_contract': 'first', 
        'commodity': 'first',  
        'min_startdate': 'min', 
        'period': ['min', 'max'] 
    }).reset_index(drop=True)

    # Rename columns 
    df_grouped.columns = ['id_debtor', 'id_contract', 'commodity', 'min_startdate', 'min_period', 'max_period']

    # Ensure the data is in the correct format and sorted
    df_grouped['min_startdate'] = pd.to_datetime(df_grouped['min_period'])
    df_grouped['max_period'] = pd.to_datetime(df_grouped['max_period'])
    df_grouped.sort_values(by=['id_debtor', 'commodity', 'id_contract', 'min_startdate'], inplace=True)
    
    # Link rows within the same id_contract
    df_grouped['prev_max_period'] = df_grouped.groupby(['id_debtor', 'commodity', 'id_contract'])['max_period'].shift(1)
    df_grouped['days_diff'] = (df_grouped['min_startdate'] - df_grouped['prev_max_period']).dt.days
    df_grouped['is_linked_within_contract'] = (df_grouped['days_diff'] <= 90) & (df_grouped['days_diff'] >= -90)
    df_grouped['contract_group_id'] = df_grouped.groupby(['id_debtor', 'commodity'])['is_linked_within_contract'].cumsum()

    # Link different contract_group_ids based on the linkage condition
    df_grouped['prev_contract_min_period'] = df_grouped.groupby(['id_debtor', 'commodity'])['min_period'].shift(1)
    df_grouped['prev_contract_max_period'] = df_grouped.groupby(['id_debtor', 'commodity'])['max_period'].shift(1)
    df_grouped['days_diff_min'] = (df_grouped['min_startdate'] - df_grouped['prev_contract_max_period']).dt.days
    df_grouped['days_diff_max'] = (df_grouped['max_period'] - df_grouped['prev_contract_min_period']).dt.days
    df_grouped['is_linked_across_contracts'] = (
        (df_grouped['days_diff_min'] <= 90) & (df_grouped['days_diff_min'] >= -90) |
        (df_grouped['days_diff_max'] <= 90) & (df_grouped['days_diff_max'] >= -90)
    )
    
    # Assign initial group IDs based on the linkage across contracts
    df_grouped['temp_id'] = (~df_grouped['is_linked_across_contracts']).cumsum()
    
    # Ensure transitivity of the ID linkage
    linked_ids = df_grouped.groupby(['id_debtor', 'commodity', 'temp_id'])['temp_id'].transform('min')
    while not (df_grouped['temp_id'] == linked_ids).all():
        df_grouped['temp_id'] = linked_ids
        linked_ids = df_grouped.groupby(['id_debtor', 'commodity', 'temp_id'])['temp_id'].transform('min')
    
    df_grouped['id'] = df_grouped['temp_id']
    
    # Remove columns used for calculations
    columns_to_drop = ['prev_max_period', 'days_diff', 'is_linked_within_contract', 'contract_group_id',
                       'prev_contract_min_period', 'prev_contract_max_period', 'days_diff_min', 'days_diff_max',
                       'is_linked_across_contracts', 'temp_id', 'linked_temp_id']
    
    columns_to_drop = [col for col in columns_to_drop if col in df_grouped.columns]
    df_grouped.drop(columns_to_drop, axis=1, inplace=True)
    
    return df_grouped

df_groups_with_id = f_id_per_group(df)

# Merge the id to the original dataframe
def f_merge_id(df, df_groups_with_id):
    df_merged = df.merge(df_groups_with_id[['id_debtor', 'id_contract', 'commodity', 'id']],
                         on=['id_debtor', 'id_contract', 'commodity'],
                         how='left')
    
    return df_merged

# Apply the function
df = f_merge_id(df, df_groups_with_id)
df = df.sort_values(by=['id', 'period'])

###### PHASE 2.4: FEATURES - CHURN INDICATOR ######

# Define a function that makes a churn indicator
def calculate_churn(df, id_column, period_column):
    df = df.copy()

    # Calculate the maximum period of the entire dataset
    overall_max_period = df[period_column].max()

    # Calculate the maximum period for each id
    max_period_per_id = df.groupby(id_column)[period_column].transform(max)

    # Initialize churn as 0 for all rows
    df['churn'] = 0

    # Set churn to 1 for the last period of an id_ean if it's not the overall max period
    df.loc[(df[period_column] == max_period_per_id) & (max_period_per_id != overall_max_period), 'churn'] = 1

    return df

# Apply the function to obtain the churn identicator
df = calculate_churn(df, 'id', 'period')

###### PHASE 2.5: FEATURES - REMAINING MONTHS IN CONTRACT ######

# Define a function that computes the remaining months in contract
# Note: this function is not correct yet
def add_remaining_months_contract(df, period_column, product_duration_column, max_end_date_column):
    df = df.copy()
    
    # Convert period and max_end_date columns to datetime
    df[period_column] = pd.to_datetime(df[period_column])
    df[max_end_date_column] = pd.to_datetime(df[max_end_date_column])
    
    # Set 'remaining_months_contract' to NA where product duration is 'UNDETERMINED'
    df['remaining_months_contract'] = pd.NA
    
    # Filter out rows where product duration is not 'UNDETERMINED'
    valid_durations = df[df[product_duration_column] != 'UNDETERMINED']
    
    # Compute the number of months between 'max_end_date' and 'period' for valid durations
    remaining_months = (
        (valid_durations[max_end_date_column] + MonthEnd(0)) - 
        (valid_durations[period_column] + MonthEnd(0))
    ).dt.days / 30.436875  # Average days per month
    
    # Round the number of months to the nearest whole number and fill in the 'remaining_months_contract'
    df.loc[valid_durations.index, 'remaining_months_contract'] = remaining_months.round().astype('Int64')
    
    return df

# Apply the function 
df = add_remaining_months_contract(df, 'period', 'productduration', 'max_enddate')

###### PHASE 2.7: FEATURES - DUMMY VARIABLES FOR SALESCHANNEL #######

# Impute own website in saleschannel 
df['saleschannel_group'] = df['saleschannel_group'].fillna('Own Website')

df_dummies = df.copy()

# Function to transform categorical column to binary columns ("other" category left out)
def f_saleschannel_dummies(df, column_name):
    # Create binary variable for each category of the specified column
    dummies = pd.get_dummies(df[column_name], dtype=int)
    
    # Easy way to create the column titles
    dummies.columns = [f"{column_name}_{col.lower().replace(' ', '').replace('/', '')}" for col in dummies.columns]
    
    # Add the dummy variables to the dataframe and drop the original categorical column
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
    
    df = df.drop('saleschannel_group_other', axis=1)
    
    return df

# Apply the function
df = f_saleschannel_dummies(df_dummies, 'saleschannel_group')

###### PHASE 2.8: OUTLIERS: NEGATIVE PRODDUCTION AND CONSUMPTION ######

# Set negative values to zero in the specified columns
columns_to_zero = ['period_consumption_low', 'period_consumption_normal',
                    'period_consumption_single', 'period_production_low',
                    'period_production_normal']

df[columns_to_zero] = df[columns_to_zero].clip(lower=0)

###### PHASE 2.9: FEATURES - PAID PRICE FOR CONSUMPTION ######

# Make paid price 
df['paid_price_consumption'] = (
    np.maximum(df['period_consumption_low'] - df['period_production_low'], 0) * df['unit_price_low'] +
    np.maximum(df['period_consumption_normal'] - df['period_production_normal'], 0) * df['unit_price_normal'] +
    np.maximum(df['period_consumption_single'] - df['period_production_single'], 0) * df['unit_price_single'] -
    (
        np.maximum(df['period_production_low'] - df['period_consumption_low'], 0) +
        np.maximum(df['period_production_normal'] - df['period_consumption_normal'], 0)
    ) * df['unit_price_feedin_surplus']
)

##### PHASE 2.10: FEATURES - CONTRACT TYPE ######

# Define a indicator variable for having a fixed contract
df['contract_type'] = np.where(df['productduration'] == 'UNDETERMINED', 0, 1)

################################### PHASE 3 ###################################

###### PHASE 3.1: LONGITUDIONAL DATA - MERGING GAS AND ELECTRICITY ######

# Create a separate data frame for electricity and gas
df_electricity = df[df['commodity'] == 'ELECTRICITY']
df_gas = df[df['commodity'] == 'GAS']

# Add indicators for gas and electricity
df_electricity['commodity_electricity'] = 1
df_gas['commodity_gas'] = 1

# Change the electricity-specific columns
electricity_columns = {
    'uuid_ean': 'uuid_ean_electricity',
    'unit_price_low': 'unit_price_low_electricity',
    'unit_price_normal': 'unit_price_normal_electricity',
    'unit_price_single': 'unit_price_single_electricity',
    'unit_price_feedin_surplus': 'unit_price_feedin_surplus_electricity',
    'annual_consumption_normal': 'annual_consumption_normal_electricity',
    'annual_production_normal': 'annual_production_normal_electricity',
    'annual_consumption_low': 'annual_consumption_low_electricity',
    'annual_production_low': 'annual_production_low_electricity',
    'annual_consumption_single': 'annual_consumption_single_electricity',
    'period_startdate': 'period_startdate_electricity',
    'period_enddate': 'period_enddate_electricity',
    'period_consumption_low': 'period_consumption_low_electricity',
    'period_consumption_normal': 'period_consumption_normal_electricity',
    'period_consumption_single': 'period_consumption_single_electricity',
    'period_production_low': 'period_production_low_electricity',
    'period_production_normal': 'period_production_normal_electricity',
    'id': 'id_electricity', 
    'churn': 'churn_electricity',
    'id_debtor': 'id_debtor_electricity',
    'id_contract': 'id_contract_electricity',
    'id_ean': 'id_ean_electricity',
    'remaining_months_contract': 'remaining_months_contract_electricity',
    'paid_price_consumption': 'paid_price_consumption_electricity', 
    'contract_type': 'contract_type_electricity',
    'fine_for_churning': 'fine_for_churning_electricity',
    'saleschannel_group_auction': 'saleschannel_auction_electricity',
    'saleschannel_group_ownwebsite': 'saleschannel_ownwebsite_electricity',
    'saleschannel_group_pricecomparisonwebsite': 'saleschannel_pricecomparisonwebsite_electricity'    # ... add other electricity-specific columns here
}
df_electricity.rename(columns=electricity_columns, inplace=True)

gas_columns = {
    'uuid_ean': 'uuid_ean_gas',
    'unit_price_low': 'unit_price_low_gas',
    'unit_price_normal': 'unit_price_normal_gas',
    'unit_price_single': 'unit_price_single_gas',
    'annual_consumption_single': 'annual_consumption_single_gas',
    'period_startdate': 'period_startdate_gas',
    'period_enddate': 'period_enddate_gas',
    'period_consumption_single': 'period_consumption_single_gas',
    'id': 'id_gas', 
    'churn': 'churn_gas',
    'id_debtor': 'id_debtor_gas',
    'id_contract': 'id_contract_gas',
    'id_ean': 'id_ean_gas',
    'remaining_months_contract': 'remaining_months_contract_gas',
    'paid_price_consumption': 'paid_price_consumption_gas', 
    'contract_type': 'contract_type_gas', 
    'fine_for_churning': 'fine_for_churning_gas',
    'saleschannel_group_auction': 'saleschannel_auction_gas',
    'saleschannel_group_ownwebsite': 'saleschannel_ownwebsite_gas',
    'saleschannel_group_pricecomparisonwebsite': 'saleschannel_pricecomparisonwebsite_gas'
}
df_gas.rename(columns=gas_columns, inplace=True)

# Define unnecessary columns
lose_columns = ['commodity', 'period_production_single']

# Drop the unnecessary columns
df_electricity = df_electricity.drop(columns=lose_columns)
df_gas = df_gas.drop(columns=lose_columns)

# Define the columns on which we merge df_gas and df_electricity
common_columns = [
    'delivery_year', 
    'delivery_month', 
    'uuid_debtornumber',
    'uuid_contractidentifier', 
    'min_startdate', 
    'max_enddate', 
    'zipcode',
    'productduration_int', 
    'productduration', 
    'period', 
    'marketingchannel_group'
]

# Merge with an outer join
df_merged = pd.merge(df_electricity, df_gas, on=common_columns, how='outer')
df_merged[['commodity_electricity', 'commodity_gas']] = df_merged[['commodity_electricity', 'commodity_gas']].fillna(0)

# Select only columns that are of float or integer type
numeric_columns = df_merged.select_dtypes(include=['float64', 'int64']).columns.tolist()
electricity_specific_columns = list(electricity_columns.values())
gas_specific_columns = list(gas_columns.values())

# Intersect with the specific columns lists to ensure we are only targeting relevant numeric columns
electricity_numeric_columns = list(set(electricity_specific_columns) & set(numeric_columns))
gas_numeric_columns = list(set(gas_specific_columns) & set(numeric_columns))

electricity_numeric_columns_without_id = [col for col in electricity_numeric_columns if 'id' not in col.lower()]
gas_numeric_columns_without_id = [col for col in gas_numeric_columns if 'id' not in col.lower()]

# Fill the relevant numeric columns with 0 for electricity and gas specific columns
for col in electricity_numeric_columns_without_id:
    df_merged.loc[df_merged['commodity_electricity'] == 0, col] = 0

for col in gas_numeric_columns_without_id:
    df_merged.loc[df_merged['commodity_gas'] == 0, col] = 0

###### PHASE 3.2: EXTRA FEATURE - ADD EXTRA FEATURE time_with_CB TO THE MERGED DF ######

# Correct function to create a feature that calculate the time a customer is with coolblue
def f_time_with_CB_feature(merged_df):
    # Ensure the 'period' and 'min_startdate' columns are in datetime format
    merged_df['period'] = pd.to_datetime(merged_df['period'])
    merged_df['min_startdate'] = pd.to_datetime(merged_df['min_startdate'])
    
    # Calculate churn end date based on churn indicators
    mask = (merged_df['churn_electricity'] == 1) & (merged_df['churn_gas'] == 1)
    merged_df.loc[mask, 'churn_end_date'] = merged_df.loc[mask, 'period']
    
    # Set churn_end_date to overall maximum period for non-churned cases
    max_period = merged_df['period'].max()
    merged_df.loc[~mask, 'churn_end_date'] = max_period
    
    # Ensure 'churn_end_date' is also in datetime format
    merged_df['churn_end_date'] = pd.to_datetime(merged_df['churn_end_date'])

    # Calculate time_with_CB in months
    merged_df['time_with_CB'] = (merged_df['churn_end_date'] - merged_df['min_startdate']).dt.days / 30.437

    # Drop the temporary column
    merged_df.drop(columns=['churn_end_date'], inplace=True)

    return merged_df

# Apply the functoin
df_merged = f_time_with_CB_feature(df_merged)

##### PHASE 3.3 FEATURES - INDICATOR POLICY CHANGE 2023-06-01 ######

# Define an indicator for the policy change in terms of new churn fined
df_merged['policy_change'] = (df_merged['period'] >= '2023-06-01').astype(int)

##### PHASE 3.4 FEATURES - INDICATOR FEW FIXED CONTRACTS 2022 ######

# Define an indicator for the period where fixed contracts were not offered
df_merged['low_offering_fixed'] = ((df_merged['period'] >= '2022-01-01') & (df_merged['period'] < '2023-02-10')).astype(int)

################################### PHASE 4 ###################################


###### PHASE 4.1: WEATHER DATA - MODIFY WEATHER DATAFRAME TO FIT OUR DATAFRAME ######

# Download the data frame
df_weather = pd.read_csv('/Users/julianjager/Library/CloudStorage/GoogleDrive-julianjager22@gmail.com/My Drive/Master/Blok 3/Seminar/Data/Externe Bronnen/Weather V2.csv')

# Make monthly data from the weather data 
def df_weather_modifier(df_weather):
       # Convert date column to datetime and set it as index
       df_weather['date'] = pd.to_datetime(df_weather['date'])
       df_weather.set_index('date', inplace=True)

       # Resample the data to get monthly averages for 'tavg', 'tmin', 'tmax', and 'prcp'
       df_weather = df_weather.resample('M')[['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd']].mean()

       # Resetting index to make date a column again
       df_weather.reset_index(inplace=True)

       # Creating separate columns for year and month
       df_weather['year'] = df_weather['date'].dt.year.astype(float).map('{:.1f}'.format)
       df_weather['month'] = df_weather['date'].dt.month.astype(float).map('{:.1f}'.format)

       # Rearranging columns for better readability
       df_weather = df_weather[['year', 'month', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd']]

       return(df_weather)

# Apply the function and delete unnecessary columns
df_weather = df_weather_modifier(df_weather)
df_weather.drop(columns=['tmin', 'tmax'], inplace=True)

###### PHASE 4.2: GAS CPI DATA - MODIFY GAS CPI DATAFRAME TO FIT OUR DATAFRAME ######

#Load the data
df_CPI_gas = pd.read_csv('/Users/julianjager/Library/CloudStorage/GoogleDrive-julianjager22@gmail.com/My Drive/Master/Blok 3/Seminar/Data/Externe Bronnen/CPI gas.csv', delimiter=';', skiprows=4)

#Function to modify the dataframe
def CPI_gas_modifier(df_CPI_gas): 
    # Reading the CSV file
    df_CPI_gas = df_CPI_gas.drop(df_CPI_gas.index[-1])
    df_CPI_gas = df_CPI_gas.drop(columns=['2015 = 100.1', '%.1'])
    df_CPI_gas = df_CPI_gas.rename(columns={'2015 = 100': 'CPI Gas (2015=100)', '%': 'Jaarmutatie CPI' })
    
    # Remove asterisks from all column names
    df_CPI_gas['Perioden'] = df_CPI_gas['Perioden'].replace('2023 december*', '2023 december')

    # Splitting the 'Perioden' column into two parts - year and month
    df_CPI_gas[['year', 'month']] = df_CPI_gas['Perioden'].str.split(' ', expand=True)

    # Mapping Dutch month names to month numbers
    months = {
        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4, 'mei': 5,
        'juni': 6, 'juli': 7, 'augustus': 8, 'september': 9,
        'oktober': 10, 'november': 11, 'december': 12
    }
    df_CPI_gas['month'] = df_CPI_gas['month'].map(months)

    # Converting the year to a numeric format
    df_CPI_gas['year'] = df_CPI_gas['year'].astype(int)
    df_CPI_gas = df_CPI_gas.drop(columns=['Perioden'])

    return (df_CPI_gas)

# Apply the function
df_CPI_gas = CPI_gas_modifier(df_CPI_gas)

###### PHASE 4.3: ELECTRICITY CPI DATA - MODIFY ELECTRICITY CPI DATAFRAME TO FIT OUR DATAFRAME ######

#Load the data
df_CPI_electricity = pd.read_csv('/Users/julianjager/Library/CloudStorage/GoogleDrive-julianjager22@gmail.com/My Drive/Master/Blok 3/Seminar/Data/Externe Bronnen/CPI electricity.csv', delimiter=';', skiprows=4)

def CPI_electricity_modifier(df_CPI_electricity): 
    # Reading the CSV file
    df_CPI_electricity = df_CPI_electricity.drop(df_CPI_electricity.index[-1])
    df_CPI_electricity = df_CPI_electricity.drop(columns=['2015 = 100.1', '%.1'])
    df_CPI_electricity = df_CPI_electricity.rename(columns={'2015 = 100': 'CPI Electricity (2015=100)', '%': 'Jaarmutatie CPI' })
    
    # Remove asterisks from all column names
    df_CPI_electricity['Perioden'] = df_CPI_electricity['Perioden'].replace('2023 december*', '2023 december')

    # Splitting the 'Perioden' column into two parts - year and month
    df_CPI_electricity[['year', 'month']] = df_CPI_electricity['Perioden'].str.split(' ', expand=True)

    # Mapping Dutch month names to month numbers
    months = {
        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4, 'mei': 5,
        'juni': 6, 'juli': 7, 'augustus': 8, 'september': 9,
        'oktober': 10, 'november': 11, 'december': 12
    }
    df_CPI_electricity['month'] = df_CPI_electricity['month'].map(months)

    # Converting the year to a numeric format
    df_CPI_electricity['year'] = df_CPI_electricity['year'].astype(int)
    df_CPI_electricity = df_CPI_electricity.drop(columns=['Perioden'])

    return (df_CPI_electricity)

# Apply the function
df_CPI_electricity = CPI_electricity_modifier(df_CPI_electricity)

###### PHASE 4.4: DAY-AHEAD PRICES DATA - MODIFY AY-AHEAD PRICES DATA TO FIT OUR DATAFRAME ######

# Download the data frame
df_prices = pd.read_csv('/Users/julianjager/Library/CloudStorage/GoogleDrive-julianjager22@gmail.com/Shared drives/Coolblue Energie Econometrics Case/Data/Day-ahead Price [EURMWh] 2018-2023.csv')

# Define a function to modify the price data frame into something more digestable
def f_day_ahead_modifier(df_prices):
    # Split 'MTU (CET/CEST)' into start and end components
    mtu_split = df_prices['MTU (CET/CEST)'].str.split(' - ', expand=True)
    start_split = mtu_split[0].str.split(' ', expand=True)
    end_split = mtu_split[1].str.split(' ', expand=True)

    # Assign components to new columns
    df_prices['start_date'] = pd.to_datetime(start_split[0], format='%d.%m.%Y')
    df_prices['start_time'] = pd.to_datetime(start_split[1], format='%H:%M').dt.time
    df_prices['end_date'] = pd.to_datetime(end_split[0], format='%d.%m.%Y')
    df_prices['end_time'] = pd.to_datetime(end_split[1], format='%H:%M').dt.time

    # Clean and convert 'Day-ahead Price [EUR/MWh]' column
    df_prices['Day-ahead Price [EUR/MWh]'] = (
        df_prices['Day-ahead Price [EUR/MWh]']
        .replace('[€,]', '', regex=True)  # Remove euro sign and commas
        .replace('', np.nan, regex=True)  # Replace empty strings with NaN
        .astype(float)  # Convert to float
    )

    # Extract year and month from start_date
    df_prices['year'] = df_prices['start_date'].dt.year
    df_prices['month'] = df_prices['start_date'].dt.month

    return df_prices

# Apply the function to the data frame
df_prices = f_day_ahead_modifier(df_prices)

# Define a data frame with only information in the 'low' hours
df_prices_low = df_prices[
    ((df_prices['start_time'] >= pd.to_datetime('23:00:00', format='%H:%M:%S').time()) |
    (df_prices['start_time'] < pd.to_datetime('07:00:00', format='%H:%M:%S').time())) &
    (df_prices['end_time'] <= pd.to_datetime('07:00:00', format='%H:%M:%S').time())
]

# Define a data frame with only information in the 'high' hours
df_prices_normal = df_prices[~df_prices.isin(df_prices_low)].dropna()
df_prices_normal.reset_index(drop=True, inplace=True)

# Convert 'year' and 'month' columns to integer type
df_prices_normal['year'] = df_prices_normal['year'].astype(int)
df_prices_normal['month'] = df_prices_normal['month'].astype(int)

#Aggregation of day-ahead prices for low, normal, and single periods
df_prices_low_aggregated = df_prices_low.groupby(['year', 'month'])['Day-ahead Price [EUR/MWh]'].agg(
    mean_low='mean',
    median_low='median',
    min_low='min',
    max_low='max',
    variance_low='var'
).reset_index()
df_prices_normal_aggregated = df_prices_normal.groupby(['year', 'month'])['Day-ahead Price [EUR/MWh]'].agg(
    mean_normal='mean',
    median_normal='median',
    min_normal='min',
    max_normal='max',
    variance_normal='var'
).reset_index()
df_prices_single_aggregated = df_prices.groupby(['year', 'month'])['Day-ahead Price [EUR/MWh]'].agg(
    mean_single='mean',
    median_single='median',
    min_single='min',
    max_single='max',
    variance_single='var'
).reset_index()

# Eventually, we will only use the df_prices_single_aggregated, with only mean and variance
df_prices_single_aggregated.drop(columns=['median_single', 'min_single', 'max_single'], inplace=True)

###### PHASE 4.5: CBS DATA - MODIY THE CBS DATA TO TO FIT OUR DATAFRAME ######

#The two dataframes (one from 2021, and one till 2021) are merged together. Also a prijsplafond dummy is added from Januari 2023. 

#Load the data
df_CBS_vanaf2021 = pd.read_csv('/Users/julianjager/Downloads/Gemiddelde_energietarieven_06022024_142757.csv', delimiter=';')
df_CBS_tot2021 = pd.read_csv('/Users/julianjager/Downloads/Gemiddelde_energietarieven__2018___2023_06022024_142850.csv', delimiter=';')

# Define a function to modify the CBS data
def f_CBS_data_modifier(df_CBS_vanaf2021, df_CBS_tot2021): 

    # Drop unneccesary columns
    df_CBS_vanaf2021.drop(['Btw', 'Aardgas/Transporttarief (Euro/jaar)','Elektriciteit/Transporttarief (Euro/jaar)', 'Aardgas/Opslag duurzame energie (ODE) (Euro/m3)', 'Elektriciteit/Opslag duurzame energie (ODE) (Euro/kWh)','Aardgas/Energiebelasting (Euro/m3)', 'Elektriciteit/Energiebelasting (Euro/kWh)', 'Aardgas/Variabel leveringstarief prijsplafond (Euro/m3)', 'Elektriciteit/Variabel leveringstarief prijsplafond (Euro/kWh)'], axis=1, inplace=True)
    df_CBS_tot2021.drop(['Btw', 'Aardgas/Transporttarief (Euro/jaar)','Elektriciteit/Transporttarief (Euro/jaar)', 'Aardgas/Opslag duurzame energie (ODE) (Euro/m3)', 'Elektriciteit/Opslag duurzame energie (ODE) (Euro/kWh)','Aardgas/Energiebelasting (Euro/m3)', 'Elektriciteit/Energiebelasting (Euro/kWh)'], axis=1, inplace=True)

    # Change the column names such that the dataframes can be merged
    df_CBS_vanaf2021 = df_CBS_vanaf2021.rename(columns={'Aardgas/Variabel leveringstarief contractprijs (Euro/m3)': 'Aardgas/Variabel leveringstarief (Euro/m3)', 'Elektriciteit/Variabel leveringstarief contractprijs (Euro/kWh)': 'Elektriciteit/Variabel leveringstarief (Euro/kWh)'})

    # Merge the dataframes together
    df_CBSdata = pd.concat([df_CBS_tot2021, df_CBS_vanaf2021], ignore_index=True)

    # Remove asterisks
    df_CBSdata['Perioden'] = df_CBSdata['Perioden'].replace('2023 december*', '2023 december')

    # Splitting the 'Perioden' column into two parts - year and month
    df_CBSdata[['year', 'month']] = df_CBSdata['Perioden'].str.split(' ', expand=True)

    # Mapping Dutch month names to month numbers
    months = {
        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4, 'mei': 5,
        'juni': 6, 'juli': 7, 'augustus': 8, 'september': 9,
        'oktober': 10, 'november': 11, 'december': 12
    }
    df_CBSdata['month'] = df_CBSdata['month'].map(months)

    # Converting the year to a numeric format
    df_CBSdata['year'] = df_CBSdata['year'].astype('Int64')
    df_CBSdata = df_CBSdata.drop(columns=['Perioden'])

    # Ddding a dummy variable 'Prijsplafond' which is equal to 1 from januari 2023. 
    df_CBSdata['Prijsplafond'] = (df_CBSdata['year'] == 2023).astype(int)

    return(df_CBSdata)

# Apply the function
df_CBSdata = f_CBS_data_modifier(df_CBS_vanaf2021, df_CBS_tot2021)

####### PHASE 4.6: MERGING EXTERNAL DATAFRAMES TO OUR MERGED_DF ######

# Function to merge the merged_df with the weather_df, CPI_electricity_df, CPI_gas_df, df_prices_low_aggregated, df_prices_normal_aggregated, df_prices_single_aggregated
def f_merging_dataframes(merged_df, df_weather, df_CPI_electricity, df_CPI_gas, df_prices_single_aggregated, df_CBSdata):
    # Ensure 'delivery_year' and delivery_month in merged_df are integers
    merged_df['delivery_year'] = pd.to_numeric(merged_df['delivery_year'], errors='coerce').astype('Int64')
    merged_df['delivery_month'] = pd.to_numeric(merged_df['delivery_month'], errors='coerce').astype('Int64')
    
    # List of dataframes to merge
    dfs_to_merge = [df_weather, df_CPI_electricity, df_CPI_gas, df_prices_low_aggregated, df_prices_normal_aggregated, df_prices_single_aggregated, df_CBSdata]
    
    # Merge the datasets
    for df in dfs_to_merge:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
        # Perform the left merge
        merged_df = pd.merge(merged_df, df, left_on=['delivery_year', 'delivery_month'], right_on=['year', 'month'], how='left').drop(columns=['year', 'month'])
    
    return merged_df

# Apply the function
df_merged_ext = f_merging_dataframes(df_merged, df_weather,df_CPI_electricity, df_CPI_gas, df_prices_single_aggregated, df_CBSdata)
df_merged_ext

# Save the dataset to a csv file
df_merged_ext.pd.to_csv('20240305_dataset_part_1.csv')


######################################################### CODE PART 2 #########################################################


############# Phase 1 - Load DataFrame #############

%reset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from scipy import stats
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

file_path = "/Users/julianjager/Desktop/20240305 Final DataFrame/20240305_dataset_part_1.csv"
df = pd.read_csv(file_path)

############# Phase 2 - Periodical Feature Engineering #############

# Create a unique_id for every customer
def f_unique_id_gas_and_electricity(df):
    df = df.copy()
    df['unique_id'] = df['id_gas'].astype(str) + '_' + df['id_electricity'].astype(str)
    df['unique_id'] = pd.factorize(df['unique_id'])[0]
    
    return df

# Apply the function
df = f_unique_id_gas_and_electricity(df)

# Drop the unnecessary columns
df = df.drop(columns = ['remaining_months_contract_gas', 'remaining_months_contract_electricity', 'time_with_CB'])

# Define a function to compute the end date
def calculate_end_date(row, min_start_date_column, product_duration_int_column):
    if row[product_duration_int_column] != 0:
        return row[min_start_date_column] + relativedelta(months=row[product_duration_int_column])
    else:
        # If product_duration_int is 0, return the min start date itself
        return row[min_start_date_column]

# Define a function to compute the remaining months in contract for gas
def add_remaining_months_contract_gas(df, period_column, min_start_date_column, product_duration_int_column):

    df = df.copy()
    
    # Convert period and min_start_date columns to datetime
    df[period_column] = pd.to_datetime(df[period_column])
    df[min_start_date_column] = pd.to_datetime(df[min_start_date_column])
    
    # Compute end date by adding product duration to min start date
    df['end_date'] = df.apply(calculate_end_date, axis=1, args=(min_start_date_column, product_duration_int_column))
    
    # Compute remaining_months_contract, setting it to 0 when product_duration_int is 0
    df['remaining_months_contract_gas'] = df.apply(
        lambda row: 0 if row[product_duration_int_column] == 0 else round(((row['end_date'] - row[period_column]).days / 30.436875)),
        axis=1
    ).astype('Int64')
    
    # Clean up by removing the end_date column
    df.drop('end_date', axis=1, inplace=True)
    
    return df

# Apply the function to get the remaining months in contract for gas
df = add_remaining_months_contract_gas(df, 'period', 'min_startdate', 'productduration_int')

# Define a function to compute the remaining months in contract for electricity
def add_remaining_months_contract_electricity(df, period_column, min_start_date_column, product_duration_int_column):

    df = df.copy()
    
    # Convert period and min_start_date columns to datetime
    df[period_column] = pd.to_datetime(df[period_column])
    df[min_start_date_column] = pd.to_datetime(df[min_start_date_column])
    
    # Compute end date by adding product duration to min start date
    df['end_date'] = df.apply(calculate_end_date, axis=1, args=(min_start_date_column, product_duration_int_column))
    
    # Compute remaining_months_contract, setting it to 0 when product_duration_int is 0
    df['remaining_months_contract_electricity'] = df.apply(
        lambda row: 0 if row[product_duration_int_column] == 0 else round(((row['end_date'] - row[period_column]).days / 30.436875)),
        axis=1
    ).astype('Int64')
    
    # Clean up by removing the end_date column
    df.drop('end_date', axis=1, inplace=True)
    
    return df

# Apply the function to get the remaining months in contract for electricity
df = add_remaining_months_contract_electricity(df, 'period', 'min_startdate', 'productduration_int')

# Define a function to get time with coolbue
def f_time_with_CB_feature(merged_df):
    # Convert 'period' to datetime format
    merged_df['period'] = pd.to_datetime(merged_df['period'])
    
    # Group by unique_id and find the minimum min_startdate for each unique_id
    min_startdate_by_id = merged_df.groupby('unique_id')['min_startdate'].min().reset_index()
    min_startdate_by_id.rename(columns={'min_startdate': 'min_startdate_global'}, inplace=True)
    
    # Merge the minimum start date back into the original DataFrame
    merged_df = merged_df.merge(min_startdate_by_id, on='unique_id', how='left')
    
    # Convert min_startdate_global to datetime format 
    merged_df['min_startdate_global'] = pd.to_datetime(merged_df['min_startdate_global'])
    
    # Calculate time_with_CB in months for each row as the difference between period and min_startdate_global
    merged_df['time_with_CB'] = (merged_df['period'] - merged_df['min_startdate_global']).dt.days / 30.437
    
    # Drop the temporary min_startdate_global column
    merged_df.drop(columns=['min_startdate_global'], inplace=True)
    
    return merged_df

# Apply the function
df = f_time_with_CB_feature(df)

############# Phase 3 - Adding seasonal features #############

# Extract month from period
df['month'] = df['period'].dt.month

# Initialize binary variables for spring, autumn, and winter
df['spring'] = 0
df['autumn'] = 0
df['winter'] = 0

# Define spring as March(3), April(4), and May(5)
df.loc[df['month'].isin([3, 4, 5]), 'spring'] = 1

# Define autumn as September(9), October(10), and November(11)
df.loc[df['month'].isin([9, 10, 11]), 'autumn'] = 1

# Define winter as December(12), January(1), and February(2)
df.loc[df['month'].isin([12, 1, 2]), 'winter'] = 1

# Drop the month column if it's no longer needed
df.drop('month', axis=1, inplace=True)

############# Phase 4 - Create Final Churn #############

# Create a function that adds churn indicators num_leads before the actual churn
def create_leading_variables(df, id_col, time_col, dependent_col, num_leads):

    # Sort the DataFrame by ID and period
    df = df.sort_values(by=[id_col, time_col])

    # Find the position of the dependent column
    col_position = df.columns.get_loc(dependent_col) + 1

    # Group by ID and create leading columns, inserting them next to the original column
    for lead in range(1, num_leads + 1):
        lead_col_name = f'{dependent_col}_lead_{lead}'
        df[lead_col_name] = df.groupby(id_col)[dependent_col].shift(-lead)
        # Move the new column to the desired position
        col_to_move = df.pop(lead_col_name)
        df.insert(col_position, lead_col_name, col_to_move)
        col_position += 1

    return df

# Apply the function for electricity and gas
df = create_leading_variables(df, 'id_electricity', 'period', 'churn_electricity', 2)
df = create_leading_variables(df, 'id_gas', 'period', 'churn_gas', 2)

# Make a final churn for electricity and gas, which is the max of the leading churns
df['final_churn_electricity'] = df[['churn_electricity', 'churn_electricity_lead_1', 'churn_electricity_lead_2']].max(axis=1)
df['final_churn_gas'] = df[['churn_gas', 'churn_gas_lead_1', 'churn_gas_lead_2']].max(axis=1)

# Make an overall churning variable, which is the max of churn in electricity and gas
df['final_churn'] = df[['final_churn_electricity', 'final_churn_gas']].max(axis=1)

############# Phase 3 - Dropping Customers without Data #############

# Check if there is data at the relevant columns
columns_to_check = [
    'period_consumption_single_electricity',
    'period_consumption_low_electricity',
    'period_consumption_normal_electricity',
    'period_consumption_single_gas',
    'period_production_low_electricity',
    'period_production_normal_electricity'
]

# Group by unique_id and check if all values are NA for the specified columns
ids_with_all_na = df.groupby('unique_id').apply(lambda group: group[columns_to_check].isna().all().all()).reset_index(name='all_na')

# Filter unique_ids where all values in the columns are NA
ids_problem = ids_with_all_na.loc[ids_with_all_na['all_na'], 'unique_id'].values

# Compute the percentage of deleted customers
len(ids_problem)/df['unique_id'].nunique()

# Drop all rows of customers without the relevant data
df_cleaned = df[~df['unique_id'].isin(ids_problem)]

############# Phase 3 - Imputing Zipcode #############

# The first option is to impute zipcode based on id and id_ean
def f_impute_zipcode_first(df, id_column, id_ean_column, zipcode_column):
    df = df.copy()

    # Group by id and id_ean, and collect unique zipcodes for each group
    grouped_zipcodes = df.groupby([id_column, id_ean_column])[zipcode_column].apply(lambda x: x.dropna().unique()).reset_index()

    # Rename the column 
    grouped_zipcodes.rename(columns={zipcode_column: 'possible_zipcodes'}, inplace=True)

    # Merge the original df with the grouped data to find possible zipcodes for each entry
    df = pd.merge(df, grouped_zipcodes, on=[id_column, id_ean_column], how='left')

    # Impute missing zipcode values where there's only one possible zipcode
    df[zipcode_column] = df.apply(
        lambda row: row['possible_zipcodes'][0] if pd.isna(row[zipcode_column]) and len(row['possible_zipcodes']) == 1 else row[zipcode_column],
        axis=1
    )

    # Drop the possible_zipcodes column as it's no longer needed
    df.drop('possible_zipcodes', axis=1, inplace=True)

    return df

# If the first option cannot find a zipcode, we impute based on the only unique zipcode for that id_debtor
def f_impute_zipcode_second(df, id_debtor_column, zipcode_column):

    df = df.copy()

    # Group by id_debtor and get unique zipcodes
    zipcodes_per_debtor = df.groupby(id_debtor_column)[zipcode_column].apply(lambda x: x.dropna().unique()).reset_index()

    # Rename the column for clarity
    zipcodes_per_debtor.rename(columns={zipcode_column: 'possible_zipcodes'}, inplace=True)

    # Filter for id_debtors that have exactly one numeric zipcode
    single_numeric_zipcode = zipcodes_per_debtor[
        zipcodes_per_debtor['possible_zipcodes'].apply(
            lambda x: len(x) == 1 and all(str(z).isnumeric() for z in x)
        )
    ]
    
    # Create a dictionary mapping id_debtor to its single numeric zipcode
    debtor_zipcode_mapping = pd.Series(single_numeric_zipcode['possible_zipcodes'].str[0].values, index=single_numeric_zipcode[id_debtor_column]).to_dict()

    # Impute the zipcode for id_debtors that have only one numeric zipcode
    df[zipcode_column] = df.apply(
        lambda row: debtor_zipcode_mapping.get(row[id_debtor_column], row[zipcode_column]) if pd.isna(row[zipcode_column]) else row[zipcode_column],
        axis=1
    )

    return df

# If the second option cannot find an unique zipcode, we impute the closest zipcode found based on period
def impute_zipcode_from_third(df, id_debtor_column, period_column, zipcode_column):

    df = df.copy()
    
    # Ensure the period_column is in datetime format
    df[period_column] = pd.to_datetime(df[period_column])
    
    # Split the DataFrame into two - one with NA zipcodes and one with known zipcodes
    df_na_zipcode = df[df[zipcode_column].isna()]
    df_known_zipcode = df[df[zipcode_column].notna()]

    # Ensure there's no empty DataFrame
    if df_na_zipcode.empty or df_known_zipcode.empty:
        return df

    # Merge the two DataFrames on id_debtor_column
    df_merged = pd.merge(df_na_zipcode, df_known_zipcode, on=id_debtor_column, suffixes=('_na', '_known'))

    # Calculate the absolute difference in periods
    df_merged['period_diff'] = (df_merged[f"{period_column}_known"] - df_merged[f"{period_column}_na"]).abs()

    # Sort by id_debtor and period difference, then drop duplicates to keep the closest period for each NA entry
    df_closest = df_merged.sort_values(by=[id_debtor_column, 'period_diff']).drop_duplicates(subset=[id_debtor_column, f"{period_column}_na"])

    # Create a dictionary to map from the id_debtor and period_na to the zipcode to be imputed
    impute_map = df_closest.set_index([id_debtor_column, f"{period_column}_na"])[zipcode_column + '_known'].to_dict()

    # Impute the zipcode
    df.loc[df[zipcode_column].isna(), zipcode_column] = df.loc[df[zipcode_column].isna()].apply(
        lambda row: impute_map.get((row[id_debtor_column], row[period_column])), axis=1)

    return df

# Apply the functions sequentially
def impute_zipcodes_sequentially(df, id_column, id_ean_column, id_debtor_column, period_column, zipcode_column):
    # Apply the first function
    df = f_impute_zipcode_first(df, id_column, id_ean_column, zipcode_column)
    
    # Check if there are still NA values in zipcode
    if df[zipcode_column].isna().sum() > 0:
        # Apply the second function
        df = f_impute_zipcode_second(df, id_debtor_column, zipcode_column)
    
    # Check again if there are still NA values in zipcode
    if df[zipcode_column].isna().sum() > 0:
        # Apply the third function
        df = impute_zipcode_from_third(df, id_debtor_column, period_column, zipcode_column)
    
    return df

# Create a unique_id for every customer
def f_unique_ean_debtor_id(df):
    df = df.copy()
    df['id_ean'] = df['uuid_ean_electricity'].astype(str) + '_' + df['uuid_ean_gas'].astype(str)
    df['id_ean'] = pd.factorize(df['id_ean'])[0]
    
    df['id_debtor'] = df['uuid_debtornumber'].astype(str)
    df['id_debtor'] = pd.factorize(df['id_debtor'])[0]
    
    return df

# Apply the function for getting ids
df_cleaned = f_unique_ean_debtor_id(df_cleaned)    
    
# Apply the function for getting zipcodes
df_cleaned = impute_zipcodes_sequentially(df_cleaned, 'unique_id', 'id_ean', 'id_debtor', 'period', 'zipcode')

############# Phase 4 - Dropping Irrelevant Columns #############

# Replace weird formats in the dataset
float_columns = ['CPI Electricity (2015=100)', 'Jaarmutatie CPI_x', 'CPI Gas (2015=100)', 'Jaarmutatie CPI_y']
df_cleaned[float_columns] = df_cleaned[float_columns].replace(',', '.', regex=True).astype(float)

# Drop columns that we do not use
columns_to_keep = [
    'period',
    'unique_id',
    'id_electricity', 
    'unit_price_low_electricity', 
    'unit_price_normal_electricity',
    'unit_price_single_electricity',
    'unit_price_feedin_surplus_electricity',
    'annual_consumption_normal_electricity',
    'annual_production_normal_electricity',
    'annual_consumption_low_electricity',
    'annual_production_low_electricity',
    'annual_consumption_single_electricity',
    'period_consumption_low_electricity',
    'period_consumption_normal_electricity',
    'period_consumption_single_electricity',
    'period_production_low_electricity',
    'period_production_normal_electricity', 
    'saleschannel_auction_electricity',
    'saleschannel_ownwebsite_electricity',
    'saleschannel_pricecomparisonwebsite_electricity',
    'zipcode', 
    'remaining_months_contract_electricity', 
    'contract_type_electricity',
    'commodity_electricity',
    'id_gas', 
    'unit_price_single_gas', 
    'annual_consumption_single_gas',
    'period_consumption_single_gas', 
    'saleschannel_auction_gas',
    'saleschannel_ownwebsite_gas',
    'saleschannel_pricecomparisonwebsite_gas',
    'remaining_months_contract_gas',
    'contract_type_gas', 
    'commodity_gas',
    'time_with_CB', 
    'policy_change', 
    'low_offering_fixed', 
    'spring',
    'autumn', 
    'winter', 
    'Prijsplafond', 
    'tavg', 
    'prcp', 
    'snow', 
    'wspd',
    'CPI Electricity (2015=100)', 
    'Jaarmutatie CPI_x', 
    'CPI Gas (2015=100)',
    'Jaarmutatie CPI_y', 
    'mean_single', 
    'variance_single',
    'Aardgas/Vast leveringstarief (Euro/jaar)',
    'Aardgas/Variabel leveringstarief (Euro/m3)',
    'Elektriciteit/Vast leveringstarief (Euro/jaar)',
    'Elektriciteit/Variabel leveringstarief (Euro/kWh)',
    'Elektriciteit/Vermindering energiebelasting (Euro/jaar)',
    'final_churn']

# Only keep the relevant columns
df_filtered = df_cleaned[columns_to_keep]

# Save the DataFrame to a csv file (data without standardization or imputation)
df_filtered.to_csv('20240305_dataset_part_2.csv', index=False)

############# Phase 6.0 - Outliers  #############

df_filtered = pd.read_csv('/Users/julianjager/Desktop/20240305 Final DataFrame/20240305_dataset_part_2.csv')

# Make sure the period is in the correct format
df_filtered['period'] = pd.to_datetime(df_filtered['period'])

monthly_dfs = {}

# Define a dataset per month
for month in range(1, 13):  
    monthly_dfs[month] = df_filtered[df_filtered['period'].dt.month == month]

# Define the columns we want to check for outliers
columns_of_interest = [
       'period_consumption_low_electricity',
       'period_consumption_normal_electricity',
       'period_consumption_single_gas',
       'period_production_low_electricity',
       'period_production_normal_electricity'
]

# Define a function to find outliers using Box-Cox transformations
def find_outliers_boxcox(df, columns_of_interest):
    outlier_indices = {}
    df_boxcox = pd.DataFrame()  
    lambda_values = {}
    
    for column in columns_of_interest:
        
        # Remove NA values and preserve the original index
        data = df[column].dropna()
        original_index = data.index
        
        # Ensure all values are positive; add a constant if 0 values are present
        data_adjusted = data + 0.01 if data.min() <= 0 else data
        
        # Apply Box-Cox transformation
        transformed_data, fitted_lambda = boxcox(data_adjusted)
        df_boxcox[column] = pd.Series(transformed_data, index=original_index) 
        lambda_values[column] = fitted_lambda
        
        # Compute Z-score: (X - mean) / std
        mean_val = np.mean(transformed_data)
        std_val = np.std(transformed_data)
        z_score = (transformed_data - mean_val) / std_val
        
        # Find indices of outliers (Z-score > 3)
        outlier_indices_temp = np.where(np.abs(z_score) > 3)[0]
        
        # Map back to original indices
        outlier_indices[column] = original_index[outlier_indices_temp].tolist()
    
    return df_boxcox, outlier_indices, lambda_values

# Find the outliers of each month
df_boxcox_1, outlier_indices_1, lambda_values_1 = find_outliers_boxcox(monthly_dfs[1], columns_of_interest)
df_boxcox_2, outlier_indices_2, lambda_values_2 = find_outliers_boxcox(monthly_dfs[2], columns_of_interest)
df_boxcox_3, outlier_indices_3, lambda_values_3 = find_outliers_boxcox(monthly_dfs[3], columns_of_interest)
df_boxcox_4, outlier_indices_4, lambda_values_4 = find_outliers_boxcox(monthly_dfs[4], columns_of_interest)
df_boxcox_5, outlier_indices_5, lambda_values_5 = find_outliers_boxcox(monthly_dfs[5], columns_of_interest)
df_boxcox_6, outlier_indices_6, lambda_values_6 = find_outliers_boxcox(monthly_dfs[6], columns_of_interest)
df_boxcox_7, outlier_indices_7, lambda_values_7 = find_outliers_boxcox(monthly_dfs[7], columns_of_interest)
df_boxcox_8, outlier_indices_8, lambda_values_8 = find_outliers_boxcox(monthly_dfs[8], columns_of_interest)
df_boxcox_9, outlier_indices_9, lambda_values_9 = find_outliers_boxcox(monthly_dfs[9], columns_of_interest)
df_boxcox_10, outlier_indices_10, lambda_values_10 = find_outliers_boxcox(monthly_dfs[10], columns_of_interest)
df_boxcox_11, outlier_indices_11, lambda_values_11 = find_outliers_boxcox(monthly_dfs[11], columns_of_interest)
df_boxcox_12, outlier_indices_12, lambda_values_12 = find_outliers_boxcox(monthly_dfs[12], columns_of_interest)

# Count the number of unique indices
unique_indices = set(index for i in range(1, 13) for indices_list in globals()[f'outlier_indices_{i}'].values() for index in indices_list)
unique_indices_list = list(unique_indices)

# Define a df with all indices that are outliers
df_outliers = df_filtered.loc[unique_indices_list]

# Count unique number of ids with outliers
id_with_outliers_count = df_outliers['unique_id'].nunique()

# Count the number of values per id
total_id_count = df_filtered['unique_id'].value_counts()

# Count the number of values per id
percentage_outliers_per_df = ((df_outliers['unique_id'].value_counts() / total_id_count) * 100).sort_values(ascending=False)

# Filter on all unique_ids with more outliers than non-outliers
outlier_ids = list(percentage_outliers_per_df[percentage_outliers_per_df > 50].index)
df_filtered = df_filtered[~df_filtered['unique_id'].isin(outlier_ids)]

############# Phase 5 - Extra Feature Engineering #############

# Define the ratio between Coolblue and the Dutch prices
df_filtered['ratio_price_cb_vs_comp_single_electricity'] = df_filtered['Elektriciteit/Variabel leveringstarief (Euro/kWh)']/df_filtered['unit_price_single_electricity']
df_filtered['ratio_price_cb_vs_comp_single_gas'] = df_filtered['Aardgas/Variabel leveringstarief (Euro/m3)']/df_filtered['unit_price_single_gas']

############# Phase 6.1 - Imputation, 0s  #############

# When customers do not have a commodity, impute zero 
def fill_na_with_zero_if_commodity_zero(df, commodity_column, columns_to_fill):
    
    # Create a mask for rows where the commodity column is 0
    mask = df[commodity_column] == 0

    # Fill NA with 0 based on the condition
    for column in columns_to_fill:
        df[column] = np.where(mask & df[column].isna(), 0, df[column])
    
    return df

# Columns to fill with 0 for gas if commodity_gas is 0
gas_columns_to_fill_with_zero = ['period_consumption_single_gas']
df_filtered = fill_na_with_zero_if_commodity_zero(df_filtered, 'commodity_gas', gas_columns_to_fill_with_zero)

# Columns to fill with 0 for electricity if commodity_electricity is 0
electricity_columns_to_fill_with_zero = [
'period_consumption_low_electricity',
 'period_consumption_normal_electricity',
 'period_production_low_electricity',
 'period_production_normal_electricity'
] 
df_filtered = fill_na_with_zero_if_commodity_zero(df_filtered, 'commodity_electricity', electricity_columns_to_fill_with_zero)     

############# Phase 6.2 - Imputation, Consumption Ratios #############

# Drop the period_consumption_single_electricity column
df_filtered = df_filtered.drop('period_consumption_single_electricity', axis=1)

# Define a function to get the Seasonal, Trend, and Residual Decomposition
def f_stl(df, column):
    
    # Ensure the period column is in datetime format
    df['period'] = pd.to_datetime(df['period'], errors='coerce')
    
    # Compute the mean per period, excluding NAs
    period_means = df.groupby('period')[column].mean().dropna()
    
    # Apply STL decomposition to the period_means
    stl = sm.tsa.STL(period_means, seasonal=13)
    result = stl.fit()
    
    return result

# Apply the function for relevant columns
df_stl_period_consumption_low_electricity = f_stl(df_filtered, 'period_consumption_low_electricity')
df_stl_period_consumption_normal_electricity = f_stl(df_filtered, 'period_consumption_normal_electricity')
df_stl_period_production_low_electricity = f_stl(df_filtered, 'period_production_low_electricity')
df_stl_period_production_normal_electricity = f_stl(df_filtered, 'period_production_normal_electricity')
df_stl_period_consumption_single_gas = f_stl(df_filtered, 'period_consumption_single_gas')

# Plot the decompositions
def plot_stl_components(stl_result, title):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=300)
    plt.suptitle(title, fontsize=14)
    
    stl_result.seasonal.plot(ax=axes[0], title='Seasonal')
    stl_result.trend.plot(ax=axes[1], title='Trend')
    stl_result.resid.plot(ax=axes[2], title='Residual')
    
    # Formatting each subplot
    for ax in axes:
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")

# Plotting STL components for each feature
features_stl = {
    'Consumption Low Electricity': df_stl_period_consumption_low_electricity,
    'Consumption Normal Electricity': df_stl_period_consumption_normal_electricity,
    'Production Low Electricity': df_stl_period_production_low_electricity,
    'Production Normal Electricity': df_stl_period_production_normal_electricity,
    'Consumption Single Gas': df_stl_period_consumption_single_gas,
}

for title, stl_result in features_stl.items():
    plot_stl_components(stl_result, title)

# Only filter on the trend and season
df_stl_period_consumption_low_electricity = df_stl_period_consumption_low_electricity.trend + df_stl_period_consumption_low_electricity.seasonal
df_stl_period_consumption_normal_electricity = df_stl_period_consumption_normal_electricity.trend + df_stl_period_consumption_normal_electricity.seasonal
df_stl_period_production_low_electricity = df_stl_period_production_low_electricity.trend + df_stl_period_production_low_electricity.seasonal
df_stl_period_production_normal_electricity = df_stl_period_production_normal_electricity.trend + df_stl_period_production_normal_electricity.seasonal
df_stl_period_consumption_single_gas = df_stl_period_consumption_single_gas.trend + df_stl_period_consumption_single_gas.seasonal

# Define a function that computes the consumption ratios of customers
def f_calculate_consumption_ratio(df, unique_id, column, stl_result):
    # Filter df for the specified unique_id and non-NA column values and calculate the mean consumption
    specific_id_data = df[(df['unique_id'] == unique_id) & df[column].notna()]
    mean_specific_consumption = specific_id_data[column].mean()
    
    if mean_specific_consumption == 0:
        return 0  # Avoid division by zero if mean consumption is 0
    
    # Determine the periods present in the filtered df
    periods_in_data = specific_id_data['period'].unique()
    
    # Filter the STL result for those periods and calculate the mean of trend + seasonal components
    # Ensure STL result is indexed by period for direct filtering
    filtered_stl_values = stl_result.loc[periods_in_data].mean()
    
    # Compute the ratio of mean specific consumption to the mean of STL (trend + seasonal)
    consumption_ratio = mean_specific_consumption / filtered_stl_values
    
    return consumption_ratio

# Define a function to loop over all customers
stl_results_dict = {
    'period_consumption_low_electricity': df_stl_period_consumption_low_electricity,
    'period_consumption_normal_electricity': df_stl_period_consumption_normal_electricity,
    'period_production_low_electricity': df_stl_period_production_low_electricity,
    'period_production_normal_electricity': df_stl_period_production_normal_electricity,
    'period_consumption_single_gas': df_stl_period_consumption_single_gas
}

# Extract all unique IDs from the DataFrame
unique_ids = df_filtered['unique_id'].unique()

# Initialize a structure to hold the results
results = []

# Define a function to loop over all customers
def process_data(df, stl_results_dict):
    # Extract all unique IDs from the DataFrame
    unique_ids = df['unique_id'].unique()
    total_ids = len(unique_ids)
    
    # Initialize a structure to hold the results
    results = []
    
    # Initialize a variable to track the next percentage threshold for printing
    next_percentage_threshold = 1

    print(f"Total unique_ids to process: {total_ids}")

    # Iterate over each unique ID
    for i, unique_id in enumerate(unique_ids):
        # Calculate the percentage done for unique_ids
        percent_done = ((i + 1) / total_ids) * 100

        # Check if the percent done has reached the next threshold
        if percent_done >= next_percentage_threshold:
            print(f"Processing unique_id {i+1}/{total_ids} ({percent_done:.1f}%)")
            next_percentage_threshold += 1 

        # Then, iterate over each feature for which we have STL results
        for column in stl_results_dict.keys():
            if i == 0: 
                print(f"Processing column: {column}")
                
            # Compute the consumption ratio for this unique_id and feature
            ratio = f_calculate_consumption_ratio(df, unique_id, column, stl_results_dict[column])
            
            # Store the result
            results.append({
                'unique_id': unique_id,
                'column': column,
                'consumption_ratio': ratio
            })

    # Convert the results list to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    return results_df

# Compute all consumption ratios
df_consumption_ratios = process_data(df_filtered, stl_results_dict)

# Put the df in a wide format
df_consumption_ratios = df_consumption_ratios.pivot(index='unique_id', columns='column', values='consumption_ratio')
df_consumption_ratios.reset_index(inplace=True)

# Save the DataFrame as a csv file
df_consumption_ratios.to_csv('20240305_consumption_ratios.csv', index=False)

############# Phase 6.2 - Imputation, Cross Validation #############

# Define a function that computes the mean percentage difference per customer
def compute_all_mean_percentage_differences(unique_id, df, stl_results_dict, df_consumption_ratios):
    mean_differences = {}

    # Iterate over each column in stl_results_dict
    for column in stl_results_dict.keys():
        # Ensure the column exists in df_consumption_ratios before proceeding
        if column in df_consumption_ratios.columns:
            # Filter df for the specified unique_id and non-NA values in the specified column
            filtered_df = df[(df['unique_id'] == unique_id) & df[column].notna()]

            # Check if all values in the column for this unique_id are 0
            if filtered_df[column].eq(0).all():
                # If all values are 0, then the mean difference is 0
                mean_differences[column] = 0
                continue  
            
            # Get all relevant periods from the filtered DataFrame
            relevant_periods = filtered_df['period']

            # Filter the corresponding STL results for these periods
            stl_values_filtered = stl_results_dict[column].reindex(relevant_periods)

            # Get the ratio for the specified unique_id and column from df_consumption_ratios
            ratio = df_consumption_ratios.loc[df_consumption_ratios['unique_id'] == unique_id, column].values[0]

            # Multiply STL values by the ratio
            adjusted_stl_values = stl_values_filtered * ratio

            # Compute the percentage difference for each period and return the mean difference
            original_values = filtered_df.set_index('period')[column].reindex(adjusted_stl_values.index).values
            percentage_differences = np.where(original_values != 0,
                                  abs(100 * (adjusted_stl_values.values - original_values) / original_values),
                                  0)

            # Return the mean of the percentage differences
            mean_difference = np.nanmean(percentage_differences)  # Use nanmean to ignore NaNs
            mean_differences[column] = abs(mean_difference)
        else:
            print(f"Column {column} not found in df_consumption_ratios.")

    return mean_differences

# Define a function that loops over all customers
def compute_percentage_differences_for_all_ids(df, stl_results_dict, df_consumption_ratios):
    all_mean_differences = {}
    unique_ids = df['unique_id'].unique()
    total_ids = len(unique_ids)
    
    last_percent_reported = -1  
    
    # Iterate through each unique_id in the dataset
    for index, unique_id in enumerate(unique_ids):
        # Calculate the current percentage of completion
        current_percent = ((index + 1) / total_ids) * 100
        if current_percent - last_percent_reported >= 1:
            print(f"Processing: {current_percent:.2f}% of unique_ids done ({index + 1}/{total_ids}).")
            last_percent_reported = current_percent
            
        # Compute mean percentage differences for the current unique_id
        mean_differences = compute_all_mean_percentage_differences(unique_id, df, stl_results_dict, df_consumption_ratios)
        
        # Store the results
        all_mean_differences[unique_id] = mean_differences
    
    print("Processing complete.")
    return all_mean_differences

# Define all the stl results
stl_results_dict = {
    'period_consumption_low_electricity': df_stl_period_consumption_low_electricity,
    'period_consumption_normal_electricity': df_stl_period_consumption_normal_electricity,
    'period_production_low_electricity': df_stl_period_production_low_electricity,
    'period_production_normal_electricity': df_stl_period_production_normal_electricity,
    'period_consumption_single_gas': df_stl_period_consumption_single_gas
}

# Apply the functions to perform cross validation
all_differences = compute_percentage_differences_for_all_ids(df_filtered, stl_results_dict, df_consumption_ratios)

# Define a function to convert the list to a dataframe
def convert_differences_to_dataframe(all_differences):
    # Flatten the all_differences dictionary
    rows_list = []
    for unique_id, columns_diff in all_differences.items():
        for column, mean_diff in columns_diff.items():
            row = {'unique_id': unique_id, 'column': column, 'mean_percentage_difference': mean_diff}
            rows_list.append(row)
    
    # Create the DataFrame
    df_all_differences = pd.DataFrame(rows_list)
    return df_all_differences

# Apply the function to get a DataFrame
df_all_differences = convert_differences_to_dataframe(all_differences)

# Use pivot table to make the wide format
df_all_differences = df_all_differences.pivot(index='unique_id', columns='column', values='mean_percentage_difference')
df_all_differences.reset_index(inplace=True)

# Save the cross validation to a csv file
df_all_differences.to_csv('20240305_cross_validation_stl.csv', index=False)

# Compute the mean deviation
df_all_differences['mean_deviation'] = df_all_differences.drop(columns=['unique_id']).replace(0, np.nan).mean(axis=1)

# Compute the number of NA's per customer
df_na_count = df_filtered.groupby('unique_id').agg(
    na_count_electricity=('period_consumption_low_electricity', lambda x: x.isna().sum()),  # Count NA values
    non_na_count_electricity=('period_consumption_low_electricity', lambda x: x.notna().sum()),  # Count non-NA values
    na_count_gas=('period_consumption_single_gas', lambda x: x.isna().sum()),  # Count NA values
    non_na_count_gas=('period_consumption_single_gas', lambda x: x.notna().sum())  # Count non-NA values
).reset_index()

# Define the final data set
df_cross_validation = pd.merge(df_all_differences, df_na_count, on='unique_id', how='inner')

# Filter only on unique_ids that will be imputed
df_cross_validation_imputation = df_cross_validation[df_cross_validation['mean_deviation'] < np.percentile(df_all_differences['mean_deviation'].replace(np.nan, 0), 95)]

############# Phase 6.3 - Imputation, Actual Imputation #############

# Define an array of ids we will delete as there is no imputation doable
cross_validation_ids = set(df_cross_validation['unique_id'])
cross_validation_imputation_ids = set(df_cross_validation_imputation['unique_id'])
ids_delete = cross_validation_ids - cross_validation_imputation_ids

# Delete all rows of these ids
df_final = df_filtered[~df_filtered['unique_id'].isin(ids_delete)]

# Make unique_id the index for easy lookup
df_consumption_ratios.set_index('unique_id', inplace=True)

# Define which columns will be imputed
columns_to_impute = [
    'period_consumption_low_electricity', 
    'period_consumption_normal_electricity', 
    'period_production_low_electricity', 
    'period_production_normal_electricity', 
    'period_consumption_single_gas'
]

# Define to print the progress
total_rows = len(df_final)
progress_interval = total_rows / 100  # 1% of total rows
next_progress_threshold = progress_interval

# Initialize a counter to track the number of processed rows
processed_rows = 0

# Loop over each row in df_cleaned
for index, row in df_final.iterrows():
    # Update the processed rows counter
    processed_rows += 1

    # For each column to impute
    for col in columns_to_impute:
        # Check if the current value is NA
        if pd.isna(row[col]):
            # Find the relevant STL DataFrame
            relevant_stl_df_name = 'df_stl_' + col
            relevant_stl_df = globals()[relevant_stl_df_name]  
            
            # Find the STL loc equal to df_cleaned[period] for the row
            period = row['period']
            stl_value = relevant_stl_df.loc[period]
            
            # Multiply with the ratio from df_consumption_ratios
            ratio = df_consumption_ratios.loc[row['unique_id']][col]
            imputed_value = stl_value * ratio
            
            # Impute the value in df_cleaned
            df_final.at[index, col] = imputed_value
    
    # Check if it's time to print progress
    if processed_rows >= next_progress_threshold:
        progress_percentage = (processed_rows / total_rows) * 100
        print(f"Imputation progress: {progress_percentage:.2f}% complete.")
        next_progress_threshold += progress_interval
        
# Make a copy of df_cleaned_final to df_imputed 
df_imputed = df_final.copy()

# Check the last columns
columns_to_check = [
    'period_consumption_low_electricity',
    'period_consumption_normal_electricity',
    'period_consumption_single_gas',
    'period_production_low_electricity',
    'period_production_normal_electricity'
]

# Fill NaN values in specified columns with 0 
for column in columns_to_check:
    df_imputed[column] = df_imputed[column].fillna(0)

# Save the DataFrame as a csv file
df_imputed.to_csv('20240305_dataset_imputed.csv', index=False)
xx = pd.read_csv('/Users/julianjager/Desktop/20240305 Final DataFrame/20240305_dataset_imputed.csv')

############# Phase 7 - External Data #############

file_path_external_data = "/Users/julianjager/Library/CloudStorage/GoogleDrive-julianjager22@gmail.com/My Drive/Master/Blok 3/Seminar/Data/Python Data 20240226/external_data_per_zip.csv"
df_external_data = pd.read_csv(file_path_external_data, sep=',')

# Create a DataFrame with unique periods
average_customer_rows = pd.DataFrame({'period': df_external_data['period'].unique()})

# Calculate mean values for each period
for col in ['HousePrice', 'percentage_residents_social_security', 'household_size', 'average_age_earner']:
    average_customer_rows[col] = average_customer_rows['period'].apply(lambda x: df_external_data.loc[df_external_data['period'] == x, col].mean())

# Add the constant values for PostalCode and Province
average_customer_rows['PostalCode'] = 10000
average_customer_rows['Province'] = np.nan

# Define the total external DataFrame
df_external_data = pd.concat([df_external_data, average_customer_rows], ignore_index=True)

# Delete the Province column
df_external_data.drop(columns=['Province'], inplace=True)

# Impute the zipcode for the normal dataset, as we use average customer inputs for these
df_not_imputed = df_filtered.copy()
df_not_imputed['zipcode'].fillna(10000, inplace=True)
df_imputed['zipcode'].fillna(10000, inplace=True)

# Make zipcode and period the same type for merging
df_imputed['zipcode'] = df_imputed['zipcode'].astype(int)
df_not_imputed['zipcode'] = df_not_imputed['zipcode'].astype(int)
df_external_data['period'] = pd.to_datetime(df_external_data['period'])

# Merge the external data to the dataset
df_imputed = pd.merge(df_imputed, df_external_data, left_on=['zipcode', 'period'], right_on=['PostalCode', 'period'], how='inner')
df_not_imputed = pd.merge(df_not_imputed, df_external_data, left_on=['zipcode', 'period'], right_on=['PostalCode', 'period'], how='inner')

# Drop zipcode
df_imputed = df_imputed.drop(columns = ['zipcode'])
df_not_imputed = df_not_imputed.drop(columns = ['zipcode'])

# Save the DataFrame as a csv file
#df_imputed.to_csv('df_imputed_after_external_data.csv', index=False)
#df_not_imputed.to_csv('df_not_imputed_after_external_data.csv', index=False)

############# Phase 8 - Unit Prizes #############

# Fill prices with 0s if a commodity identifyer is 0
def fill_na_with_zeros(df, commodity_column, consumption_columns):

    # Identify rows where the commodity is 0 and there's an NA in any of the consumption columns
    mask = (df[commodity_column] == 0) & df[consumption_columns].isna().any(axis=1)

    # Fill NA for rows where commodity is 0 and there's an NA in the consumption columns with 0's
    df.loc[mask, consumption_columns] = df.loc[mask, consumption_columns].fillna(0)

    return df

# Fill NA for gas
gas_consumption_columns = ['unit_price_single_gas']

df_imputed = fill_na_with_zeros(df_imputed, 'commodity_gas', gas_consumption_columns)
df_not_imputed = fill_na_with_zeros(df_not_imputed, 'commodity_gas', gas_consumption_columns)

# Fill NA for electricity
electricity_price_columns = [
    'unit_price_low_electricity',
    'unit_price_normal_electricity',
    'unit_price_single_electricity',
    'unit_price_feedin_surplus_electricity'
]

df_imputed = fill_na_with_zeros(df_imputed, 'commodity_electricity', electricity_price_columns)
df_not_imputed = fill_na_with_zeros(df_not_imputed, 'commodity_electricity', electricity_price_columns)

# Redefine the ratios of the price of Coolblue compared to competitors
df_imputed['ratio_price_cb_vs_comp_single_electricity'] = df_imputed['Elektriciteit/Variabel leveringstarief (Euro/kWh)']/df_imputed['unit_price_single_electricity']
df_imputed['ratio_price_cb_vs_comp_single_gas'] = df_imputed['Aardgas/Variabel leveringstarief (Euro/m3)']/df_imputed['unit_price_single_gas']
df_not_imputed['ratio_price_cb_vs_comp_single_electricity'] = df_not_imputed['Elektriciteit/Variabel leveringstarief (Euro/kWh)']/df_not_imputed['unit_price_single_electricity']
df_not_imputed['ratio_price_cb_vs_comp_single_gas'] = df_not_imputed['Aardgas/Variabel leveringstarief (Euro/m3)']/df_not_imputed['unit_price_single_gas']

# Drop PostalCode and Province
df_imputed = df_imputed.drop(columns = ['PostalCode'])
df_not_imputed = df_not_imputed.drop(columns = ['PostalCode'])

############# Phase 9 - Impute 0s for Variable Contracts #############

# Define DataFrames for the machine learning models
#df_rsf_xgb = df_imputed_not_standardized.copy()
#df_lr_lstm = df_imputed_standardized.copy()

# Impute 0s when there is is a variable contract
df_imputed['remaining_months_contract_gas'] = df_imputed['remaining_months_contract_gas'].fillna(0)
df_imputed['remaining_months_contract_electricity'] = df_imputed['remaining_months_contract_electricity'].fillna(0)
df_not_imputed['remaining_months_contract_gas'] = df_not_imputed['remaining_months_contract_gas'].fillna(0)
df_not_imputed['remaining_months_contract_electricity'] = df_not_imputed['remaining_months_contract_electricity'].fillna(0)

# Save datasets without the N/As
#df_rsf_xgb.to_csv('df_rsf_xbg_no_NAs.csv', index=False)
#df_lr_lstm.to_csv('df_lr_lstm_no_NAs.csv', index=False)

############# Phase 11 - Variable for Paid Price #############

# Add the paid price to the df for electricity
def calculate_paid_price_consumption_electricity(df):
    df['paid_price_consumption_electricity'] = (
        np.maximum(df['period_consumption_low_electricity'] - df['period_production_low_electricity'], 0) * df['unit_price_low_electricity'] +
        np.maximum(df['period_consumption_normal_electricity'] - df['period_production_normal_electricity'], 0) * df['unit_price_normal_electricity'] -
        (
            np.maximum(df['period_production_low_electricity'] - df['period_consumption_low_electricity'], 0) +
            np.maximum(df['period_production_normal_electricity'] - df['period_consumption_normal_electricity'], 0)
        ) * df['unit_price_feedin_surplus_electricity']
    )
    return df

# Add the paid price to the df for gas
def calculate_paid_price_consumption_gas(df):
    df['paid_price_consumption_gas'] = df['period_consumption_single_gas'] * df['unit_price_single_gas']
    return df

# Apply the functions for RSF and XGBoost
df_imputed = calculate_paid_price_consumption_electricity(df_imputed)
df_imputed = calculate_paid_price_consumption_gas(df_imputed)

# Apply the functions for LR and LSTM
df_not_imputed = calculate_paid_price_consumption_electricity(df_not_imputed)
df_not_imputed = calculate_paid_price_consumption_gas(df_not_imputed)

############# Phase 12 - NA Check #############

# List of columns to check for NAs
columns_to_check = [
 'annual_consumption_normal_electricity',
       'annual_production_normal_electricity',
       'annual_consumption_low_electricity',
       'annual_production_low_electricity',
       'annual_consumption_single_electricity',
       'period_consumption_low_electricity',
       'period_consumption_normal_electricity',
       'period_production_low_electricity',
       'period_production_normal_electricity',
       'annual_consumption_single_gas',
       'period_consumption_single_gas'
]

df_imputed[df_imputed[columns_to_check].isna().any(axis=1)]['unique_id'].unique()
df_not_imputed[df_not_imputed[columns_to_check].isna().any(axis=1)]['unique_id'].unique()

############# Phase 13 - Lagged Variables #############

#file_path_df_rsf_xgb_pp = "/Users/julianjager/df_rsf_xbg_pp.csv"
#file_path_df_lr_lstm_pp = "/Users/julianjager/df_lr_lstm_pp.csv"

#df_rsf_xgb_lagged = pd.read_csv(file_path_df_rsf_xgb_pp)
#df_lr_lstm_lagged = pd.read_csv(file_path_df_lr_lstm_pp)

#df_rsf_xgb_lagged = df_rsf_xgb_pp.copy()
#df_lr_lstm_lagged = df_lr_lstm_pp.copy()

df_imputed_not_lagged = df_imputed.copy()
df_not_imputed_not_lagged = df_not_imputed.copy()
df_imputed_lagged = df_imputed.copy()
df_not_imputed_lagged = df_not_imputed.copy()

# Define a function to create lagged variables
def create_lagged_variables(df, id_col, time_col, dependent_col, num_lags):

    # Sort the DataFrame by ID and time
    df = df.sort_values(by=[id_col, time_col])

    # Find the position of the dependent column
    col_position = df.columns.get_loc(dependent_col) + 1

    # Group by ID and create lagged columns, inserting them next to the original column
    for lag in range(1, num_lags + 1):
        lag_col_name = f'{dependent_col}_lag_{lag}'
        df[lag_col_name] = df.groupby(id_col)[dependent_col].shift(lag)
        # Move the new column next to the other column for readability
        col_to_move = df.pop(lag_col_name)
        df.insert(col_position, lag_col_name, col_to_move)
        col_position += 1

    return df

df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_electricity', 'period', 'period_consumption_low_electricity', 11)
df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_electricity', 'period', 'period_consumption_normal_electricity', 11)
df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_electricity', 'period', 'period_production_low_electricity', 11)
df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_electricity', 'period', 'period_production_normal_electricity', 11)
df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_electricity', 'period', 'paid_price_consumption_electricity', 11)
df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_gas', 'period', 'period_consumption_single_gas', 11)
df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'unique_id', 'period', 'paid_price_consumption', 11)
#df_imputed_lagged = create_lagged_variables(df_imputed_lagged, 'id_gas', 'period', 'paid_price_consumption_gas', 11)

df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_electricity', 'period', 'period_consumption_low_electricity', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_electricity', 'period', 'period_consumption_normal_electricity', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_electricity', 'period', 'period_production_low_electricity', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_electricity', 'period', 'period_production_normal_electricity', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_electricity', 'period', 'paid_price_consumption_electricity', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_gas', 'period', 'period_consumption_single_gas', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_gas', 'period', 'paid_price_consumption_electricity', 11)
df_not_imputed_lagged = create_lagged_variables(df_not_imputed_lagged, 'id_gas', 'period', 'paid_price_consumption_gas', 11)

############# Phase 14 - Aggregation #############

#df_lr = df_imputed_lagged.copy()
#df_rsf_xgb = df_imputed_lagged.copy()
df_imputed_agg = df_imputed_lagged.copy()

# Function to process the variables with the first 6 columns as t to t-5, and the last 6 columns as t-6 to t-11
def f_lagged_column_names(col_name):
    first_6_columns = [col_name] + [f'{col_name}_lag_{i}' for i in range(1, 6)]
    last_6_columns = [f'{col_name}_lag_{i}' for i in range(6, 12)]
    return first_6_columns, last_6_columns

# Create the names of the features
first_6_period_consumption_low_electricity, last_6_period_consumption_low_electricity = f_lagged_column_names('period_consumption_low_electricity')
first_6_period_consumption_normal_electricity, last_6_period_consumption_normal_electricity = f_lagged_column_names('period_consumption_normal_electricity')
first_6_period_period_production_low_electricity, last_6_period_production_low_electricity = f_lagged_column_names('period_production_low_electricity')
first_6_period_production_normal_electricity, last_6_period_production_normal_electricity = f_lagged_column_names('period_production_normal_electricity')
first_6_period_consumption_single_gas, last_6_period_consumption_single_gas = f_lagged_column_names('period_consumption_single_gas')
first_6_paid_price_consumption, last_6_paid_price_consumption = f_lagged_column_names('paid_price_consumption')
#first_6_paid_price_consumption_electricity, last_6_paid_price_consumption_electricity = f_lagged_column_names('paid_price_consumption_electricity')
#first_6_paid_price_consumption_gas, last_6_paid_price_consumption_gas = f_lagged_column_names('paid_price_consumption_gas')

# Create the aggregated variables
def process_variables(df, first_6_columns, last_6_columns):
    # Get the base name from the first of the first_6_columns
    base_name = first_6_columns[0]
    
    # Calculate the mean of the first 6 variables for each row
    df[f'{base_name}_mean_of_first_6'] = df[first_6_columns].mean(axis=1)
    
    # Calculate the difference between the value at t and value at t-5, then divide by 6
    # df[f'{base_name}_avg_diff_t_to_t-5'] = (df[first_6_columns[0]] - df[first_6_columns[5]]) / 6
    
    # Calculate the normalized value for each of t to t-5
    # by the mean value of the last 6 variables of that row
    #for i, column in enumerate(first_6_columns):
        #df[f'{base_name}_normalized_{column}'] = df[column] / df[last_6_columns].mean(axis=1)
    
    return df

# Apply the function
df_imputed_agg = process_variables(df_imputed_agg, first_6_period_consumption_low_electricity, last_6_period_consumption_normal_electricity)
df_imputed_agg = process_variables(df_imputed_agg, first_6_period_consumption_normal_electricity, last_6_period_consumption_normal_electricity)
df_imputed_agg = process_variables(df_imputed_agg, first_6_period_period_production_low_electricity, last_6_period_production_low_electricity)
df_imputed_agg = process_variables(df_imputed_agg, first_6_period_production_normal_electricity, last_6_period_production_normal_electricity)
df_imputed_agg = process_variables(df_imputed_agg, first_6_period_consumption_single_gas, last_6_period_consumption_single_gas)
df_imputed_agg = process_variables(df_imputed_agg, first_6_paid_price_consumption, last_6_paid_price_consumption)
#df_imputed_agg = process_variables(df_imputed_agg, first_6_paid_price_consumption_electricity, last_6_paid_price_consumption_electricity)
#df_imputed_agg = process_variables(df_imputed_agg, first_6_paid_price_consumption_gas, last_6_paid_price_consumption_gas)

df_not_imputed_agg = process_variables(df_not_imputed_lagged, first_6_period_consumption_low_electricity, last_6_period_consumption_normal_electricity)
df_not_imputed_agg = process_variables(df_not_imputed_agg, first_6_period_consumption_normal_electricity, last_6_period_consumption_normal_electricity)
df_not_imputed_agg = process_variables(df_not_imputed_agg, first_6_period_period_production_low_electricity, last_6_period_production_low_electricity)
df_not_imputed_agg = process_variables(df_not_imputed_agg, first_6_period_production_normal_electricity, last_6_period_production_normal_electricity)
df_not_imputed_agg = process_variables(df_not_imputed_agg, first_6_period_consumption_single_gas, last_6_period_consumption_single_gas)
df_not_imputed_agg = process_variables(df_not_imputed_agg, first_6_paid_price_consumption_electricity, last_6_paid_price_consumption_electricity)
df_not_imputed_agg = process_variables(df_not_imputed_agg, first_6_paid_price_consumption_gas, last_6_paid_price_consumption_gas)

############# Phase 15 - Delete Columns #############

# We now have
df_imputed_agg # for lr/rsf/xgb
df_not_imputed_agg # for xgb
df_imputed # for lstm

# Delete other variables
columns_to_delete = [
    'period_consumption_low_electricity', 'period_consumption_normal_electricity',
    'period_production_low_electricity', 'period_production_normal_electricity', 'period_consumption_single_gas',
    'paid_price_consumption_electricity', 'paid_price_consumption_gas'
]

# Delete first and last 6 period columns
for col_name in columns_to_delete:
    df_imputed_agg.drop(columns=[f'{col_name}_lag_{i}' for i in range(1, 12)], inplace=True)
    
for col_name in columns_to_delete:
    df_not_imputed_agg.drop(columns=[f'{col_name}_lag_{i}' for i in range(1, 12)], inplace=True)

# Check which columns have NAs
df_imputed_agg.columns[df_imputed_agg.isna().any()].tolist()

# Impute period averages
def impute_with_period_average(df):
    
    # Group by period and calculate the average of each specified column
    averages_per_period = df.groupby('period')[['percentage_residents_social_security', 'household_size', 'average_age_earner']].transform('mean')

    # Impute missing values in each column with the corresponding period average
    df['percentage_residents_social_security'].fillna(averages_per_period['percentage_residents_social_security'], inplace=True)
    df['household_size'].fillna(averages_per_period['household_size'], inplace=True)
    df['average_age_earner'].fillna(averages_per_period['average_age_earner'], inplace=True)
    df['snow'].fillna(0, inplace=True)
    
    return df

df_imputed = impute_with_period_average(df_imputed) # for lstm
df_imputed_agg = impute_with_period_average(df_imputed_agg) # for lr/rsf/xgb
df_not_imputed_agg # for xgb

############# Phase 17 - New Variables #############

# Make a remaining months in contract variable
df_imputed['remaining_months_contract'] = df_imputed['remaining_months_contract_gas']
df_imputed_agg['remaining_months_contract'] = df_imputed_agg['remaining_months_contract_gas']
df_not_imputed_agg['remaining_months_contract'] = df_not_imputed_agg['remaining_months_contract_gas']

# Make fine for churning
def calculate_max_fine_for_churning(df):
    # Define fines for churning based on conditions
    df['fine_for_churning_electricity'] = np.where(
        (df['remaining_months_contract'] > 1) & (df['contract_type_electricity'] == 1), 
        1, 
        0
    )
    df['fine_for_churning_gas'] = np.where(
        (df['remaining_months_contract'] > 1) & (df['contract_type_gas'] == 1), 
        1, 
        0
    )
    
    # Calculate max fine for churning between gas and electricity
    df['max_fine_for_churning'] = df[['fine_for_churning_electricity', 'fine_for_churning_gas']].max(axis=1)
    
    return df

# Apply the function
df_imputed = calculate_max_fine_for_churning(df_imputed)
df_imputed_agg = calculate_max_fine_for_churning(df_imputed_agg)
df_not_imputed_agg = calculate_max_fine_for_churning(df_not_imputed_agg)

# Define a function to make a solar panel indicator
def make_solar_panel_ind_modified(df):
    
    # Function to apply to each group
    def set_solar_panels(group):
        # Find the first period where the condition is met
        condition_met = (group['period_production_low_electricity'] + 
                         group['period_production_normal_electricity'] > 0).idxmax()
        # Set indicator based on the condition
        group['solar_panels'] = 0
        if group.loc[condition_met, 'period_production_low_electricity'] + group.loc[condition_met, 'period_production_normal_electricity'] > 0:
            group.loc[condition_met:, 'solar_panels'] = 1
        return group

    # Sort by unique_id and period to ensure chronological order
    df = df.sort_values(by=['unique_id', 'period'])
    
    # Apply the function to each group
    df = df.groupby('unique_id').apply(set_solar_panels)
    
    return df

# Apply the function
df_imputed = make_solar_panel_ind_modified(df_imputed)
df_imputed_agg = make_solar_panel_ind_modified(df_imputed_agg)
df_not_imputed_agg = make_solar_panel_ind_modified(df_not_imputed_agg)


############# Phase 17 - Making columns together #############

# Transform some columna that are already made
def transform_df(df):

    # Make paid_price_consumption
    df['paid_price_consumption'] = df['paid_price_consumption_electricity'] + df['paid_price_consumption_gas']

    # Make saleschannel_auction
    df['saleschannel_auction'] = df[['saleschannel_auction_electricity', 'saleschannel_auction_gas']].max(axis=1)

    # Make saleschannel_ownwebsite
    df['saleschannel_ownwebsite'] = df[['saleschannel_ownwebsite_electricity', 'saleschannel_ownwebsite_gas']].max(axis=1)

    # Make saleschannel_pricecomparisonwebsite
    df['saleschannel_pricecomparisonwebsite'] = df[['saleschannel_pricecomparisonwebsite_electricity', 'saleschannel_pricecomparisonwebsite_gas']].max(axis=1)

    return df

# Apply the function
df_imputed = transform_df(df_imputed)
df_imputed_agg = transform_df(df_imputed_agg)
df_not_imputed_agg = transform_df(df_not_imputed_agg)
df_imputed_lagged = transform_df(df_imputed_lagged)


############# Phase 17 - Deleting Columns #############


# Drop unnecessary columns
columns_to_drop = [
    "Aardgas/Variabel leveringstarief (Euro/m3)",
    "Aardgas/Vast leveringstarief (Euro/jaar)",
    "Elektriciteit/Variabel leveringstarief (Euro/kWh)",
    "Elektriciteit/Vast leveringstarief (Euro/jaar)",
    "fine_for_churning_electricity",
    "fine_for_churning_gas",
    "id_electricity",
    "id_gas",
    "Jaarmutatie CPI_x",
    "Jaarmutatie CPI_y",
    "paid_price_consumption_electricity",
    "paid_price_consumption_gas",
    "remaining_months_contract_electricity",
    "remaining_months_contract_gas",
    "saleschannel_auction_electricity",
    "saleschannel_auction_gas",
    "saleschannel_ownwebsite_electricity",
    "saleschannel_ownwebsite_gas",
    "saleschannel_pricecomparisonwebsite_electricity",
    "saleschannel_pricecomparisonwebsite_gas"
]

df_imputed = df_imputed.drop(columns=columns_to_drop)
df_imputed_agg = df_imputed_agg.drop(columns=columns_to_drop)
df_not_imputed_agg = df_not_imputed_agg.drop(columns=columns_to_drop)


############# Phase 17 - Renaming Columns #############

rename_columns = {
    'period': 'period',
    'unique_id': 'unique_id',
    'unit_price_low_electricity': 'unit_price_low_electricity',
    'unit_price_normal_electricity': 'unit_price_normal_electricity',
    'unit_price_single_electricity': 'unit_price_single_electricity',
    'unit_price_feedin_surplus_electricity': 'unit_price_feedin_surplus_electricity',
    'annual_consumption_normal_electricity': 'annual_consumption_normal_electricity',
    'annual_production_normal_electricity': 'annual_production_normal_electricity',
    'annual_consumption_low_electricity': 'annual_consumption_low_electricity',
    'annual_production_low_electricity': 'annual_production_low_electricity',
    'annual_consumption_single_electricity': 'annual_consumption_single_electricity',
    'period_consumption_low_electricity': 'period_consumption_low_electricity',
    'period_consumption_normal_electricity': 'period_consumption_normal_electricity',
    'period_production_low_electricity': 'period_production_low_electricity',
    'period_production_normal_electricity': 'period_production_normal_electricity',
    'contract_type_electricity': 'fixed_contract_electricity_ind',
    'commodity_electricity': 'electricity_ind',
    'unit_price_single_gas': 'unit_price_single_gas',
    'annual_consumption_single_gas': 'annual_consumption_single_gas',
    'period_consumption_single_gas': 'period_consumption_single_gas',
    'contract_type_gas': 'fixed_contract_gas_ind',
    'commodity_gas': 'gas_ind',
    'time_with_CB': 'time_with_coolblue',
    'policy_change': 'changed_policy_ind',
    'low_offering_fixed': 'no_fixed_contracts_offered_ind',
    'spring': 'spring_ind',
    'autumn': 'autumn_ind',
    'winter': 'winter_ind',
    'Prijsplafond': 'price_cap_ind',
    'tavg': 'average_temperature',
    'prcp': 'precipitation',
    'snow': 'snow',
    'wspd': 'windspeed',
    'CPI Electricity (2015=100)': 'cpi_electricity',
    'CPI Gas (2015=100)': 'cpi_gas',
    'mean_single': 'mean_day_ahead_price_electricity',
    'variance_single': 'variance_day_ahead_price_electricity',
    'Elektriciteit/Vermindering energiebelasting (Euro/jaar)': 'reduction_energy_taxes',
    'final_churn': 'final_churn',
    'ratio_price_cb_vs_comp_single_electricity': 'ratio_price_cb_vs_comp_electricity',
    'ratio_price_cb_vs_comp_single_gas': 'ratio_price_cb_vs_comp_gas',
    'HousePrice': 'house_price',
    'percentage_residents_social_security': 'percentage_residents_social_security',
    'household_size': 'household_size',
    'average_age_earner': 'average_age_earner',
    'remaining_months_contract': 'remaining_months_contract',
    'max_fine_for_churning': 'fine_for_churning_ind',
    'solar_panels': 'solar_panels_ind',
    'paid_price_consumption': 'paid_price_consumption',
    'saleschannel_auction': 'saleschannel_auction_ind',
    'saleschannel_ownwebsite': 'saleschannel_ownwebsite_ind',
    'saleschannel_pricecomparisonwebsite': 'saleschannel_pricecomparisonwebsite_ind'
}

df_imputed = df_imputed.rename(columns=rename_columns)
df_imputed_agg = df_imputed_agg.rename(columns=rename_columns)
df_not_imputed_agg = df_not_imputed_agg.rename(columns=rename_columns)

df_imputed.to_csv('20240305_df_imputed.csv', index=False)
df_imputed_agg.to_csv('20240305_df_imputed_agg.csv', index=False)
df_not_imputed_agg.to_csv('20240305_df_not_imputed_agg.csv', index=False)

df_imputed = pd.read_csv('/Users/julianjager/Desktop/20240305 Final DataFrame/20240305_df_imputed.csv')
df_imputed_agg = pd.read_csv('/Users/julianjager/Desktop/20240305 Final DataFrame/20240305_df_imputed_agg.csv')
df_not_imputed_agg = pd.read_csv('/Users/julianjager/Desktop/20240305 Final DataFrame/20240305_df_not_imputed_agg.csv')


############# COPYING DATAFRAMES #############

df_lstm = df_imputed.copy()
df_lr = df_imputed_agg.copy()
df_xgb = df_imputed_agg.copy()
df_rsf = df_imputed_agg.copy()

############# STANDARDIZATION #############

# Apply standardization

# First replace inf values
max_value = df_imputed['ratio_price_cb_vs_comp_electricity'].max()
if np.isinf(max_value):
    mean_value = df_imputed['ratio_price_cb_vs_comp_electricity'].replace(np.inf, np.nan).mean()
    df_imputed.loc[df_imputed['ratio_price_cb_vs_comp_electricity'] == max_value, 'ratio_price_cb_vs_comp_electricity'] = mean_value
max_value = df_imputed['ratio_price_cb_vs_comp_gas'].max()
if np.isinf(max_value):
    mean_value = df_imputed['ratio_price_cb_vs_comp_gas'].replace(np.inf, np.nan).mean()
    df_imputed.loc[df_imputed['ratio_price_cb_vs_comp_gas'] == max_value, 'ratio_price_cb_vs_comp_gas'] = mean_value
max_value = df_imputed_agg['ratio_price_cb_vs_comp_electricity'].max()
if np.isinf(max_value):
    mean_value = df_imputed_agg['ratio_price_cb_vs_comp_electricity'].replace(np.inf, np.nan).mean()
    df_imputed_agg.loc[df_imputed_agg['ratio_price_cb_vs_comp_electricity'] == max_value, 'ratio_price_cb_vs_comp_electricity'] = mean_value
max_value = df_imputed_agg['ratio_price_cb_vs_comp_gas'].max()
if np.isinf(max_value):
    mean_value = df_imputed_agg['ratio_price_cb_vs_comp_gas'].replace(np.inf, np.nan).mean()
    df_imputed_agg.loc[df_imputed_agg['ratio_price_cb_vs_comp_gas'] == max_value, 'ratio_price_cb_vs_comp_gas'] = mean_value

# Define a function for standardization
def f_standardize(data, columns_non_standardize):
    
    # Select columns to standardize
    columns_to_standardize = data.columns.difference(columns_non_standardize)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit on data and transform it for selected columns
    data_standardized = data.copy()
    data_standardized[columns_to_standardize] = scaler.fit_transform(data_standardized[columns_to_standardize])

    return data_standardized

# Define which columns does not need standardization
non_standardized_columns = ['unique_id',
                            'period',
                            'fixed_contract_electricity_ind',
                            'electricity_ind',
                            'fixed_contract_gas_ind',
                            'gas_ind',
                            'changed_policy_ind',
                            'no_fixed_contracts_offered_ind', 
                            'spring_ind', 
                            'autumn_ind',
                            'winter_ind', 
                            'price_cap_ind', 
                            'final_churn', 
                            'fine_for_churning_ind', 
                            'solar_panels_ind', 
                            'saleschannel_auction_ind', 
                            'saleschannel_ownwebsite_ind',
                            'saleschannel_pricecomparisonwebsite_ind']

# Define all DataFrames 
df_lstm = f_standardize(df_imputed, non_standardized_columns)
df_lr = f_standardize(df_imputed_agg, non_standardized_columns)
df_rsf = df_imputed_agg.copy()
df_xgb = df_imputed_agg.copy()

# Save the final datasets
df_lstm.to_csv('20240305_lstm.csv', index=False)
df_lr.to_csv('20240307_lr.csv', index=False)
df_rsf.to_csv('20240307_rsf.csv', index=False)
df_xgb.to_csv('20240307_xgb.csv', index=False)

df_lr['annual_consumption_single_electricity']