import pandas as pd

storm_df = pd.read_csv('/kaggle/input/noaa-powout-prism-0-1/noaapowoutprism_01.csv')

# Ensure power_outage_datetime is in datetime format
storm_df['power_outage_datetime'] = pd.to_datetime(storm_df['power_outage_datetime'])

# Sort the DataFrame by state and datetime to ensure temporal order
storm_df = storm_df.sort_values(by=['STATE_FIPS', 'power_outage_datetime'])

# Function to apply a 1-hour lag within each state group
def apply_1hour_lag(group):
    # Initialize the lagged column
    group['is_storm_lagged'] = None
    
    # Iterate through the group to find the appropriate row for 1-hour lag
    for i in range(len(group)):
        current_time = group.iloc[i]['power_outage_datetime']
        future_time = current_time + pd.Timedelta(hours=1)
        
        # Find the closest row within the same state that is at least 1 hour ahead
        future_rows = group[(group['power_outage_datetime'] >= future_time) & 
                           (group['power_outage_datetime'] <= future_time + pd.Timedelta(hours=1))]
        
        if not future_rows.empty:
            # Take the first row (closest to 1 hour)
            group.iloc[i, group.columns.get_loc('is_storm_lagged')] = future_rows.iloc[0]['is_storm']
    
    return group

# Apply the lagging function to each state group
storm_df = storm_df.groupby('STATE_FIPS').apply(apply_1hour_lag).reset_index(drop=True)

# Drop rows where is_storm_lagged is null
storm_df = storm_df.dropna(subset=['is_storm_lagged'])

# Display the result
print(storm_df[['power_outage_datetime', 'STATE_FIPS', 'is_storm', 'is_storm_lagged']].head(10))