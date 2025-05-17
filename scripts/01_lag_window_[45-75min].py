import pandas as pd
import numpy as np
from tqdm import tqdm

storm_df = pd.read_csv('/kaggle/input/noaa-powout-prism-0-1/noaapowoutprism_01.csv')

# Enable tqdm for pandas apply
tqdm.pandas()

# Assuming storm_df is already loaded
# Ensure datetime columns are in datetime format
storm_df['power_outage_datetime'] = pd.to_datetime(storm_df['power_outage_datetime'])
storm_df['storm_start_datetime_est'] = pd.to_datetime(storm_df['storm_start_datetime_est'])
storm_df['storm_end_datetime_est'] = pd.to_datetime(storm_df['storm_end_datetime_est'])

# Sort by state and datetime to ensure temporal order
storm_df = storm_df.sort_values(['st_abb', 'power_outage_datetime']).reset_index(drop=True)

# Step 1: Create IS_STORM_1HR_AHEAD column with ±15-minute buffer
def check_storm_1hr_ahead(row, df, buffer_minutes=15):
    current_time = row['power_outage_datetime']
    state = row['st_abb']
    # Define 1-hour window with ±15-minute buffer (45 min to 1 hr 15 min)
    start_time = current_time + pd.Timedelta(minutes=45)
    end_time = current_time + pd.Timedelta(minutes=75)
    # Filter records for the same state where either start or end time is in the window
    future_records = df[
        (df['st_abb'] == state) &
        (
            ((df['storm_start_datetime_est'] >= start_time) & (df['storm_start_datetime_est'] <= end_time)) |
            ((df['storm_end_datetime_est'] >= start_time) & (df['storm_end_datetime_est'] <= end_time))
        )
    ]
    if future_records.empty:
        return None  # Mark for removal
    # Check if any record has is_storm == 1
    return int(future_records['is_storm'].eq(1).any())

# Apply the function using tqdm for progress visualization
print("Calculating IS_STORM_1HR_AHEAD...")
storm_df['IS_STORM_1HR_AHEAD'] = storm_df.progress_apply(
    lambda row: check_storm_1hr_ahead(row, storm_df, buffer_minutes=15), axis=1
)

# Remove rows where no records exist (IS_STORM_1HR_AHEAD is None)
print(f"Original rows: {len(storm_df)}")
storm_df = storm_df[storm_df['IS_STORM_1HR_AHEAD'].notnull()].reset_index(drop=True)
storm_df['IS_STORM_1HR_AHEAD'] = storm_df['IS_STORM_1HR_AHEAD'].astype(int)
print(f"Rows after removing no-record cases: {len(storm_df)}")

# # Step 2: Encode categorical variables
# if 'EVENT_TYPE_encoded' not in storm_df.columns:
#     le_event_type = LabelEncoder()
#     storm_df['EVENT_TYPE_encoded'] = le_event_type.fit_transform(storm_df['EVENT_TYPE'])

# if 'stability_encoded' not in storm_df.columns:
#     le_stability = LabelEncoder()
#     storm_df['stability_encoded'] = le_stability.fit_transform(storm_df['stability'])

# Step 3: Export the updated dataframe as CSV
output_filename = "lagged_data.csv"
storm_df.to_csv(output_filename, index=False)
print(f"CSV exported: {output_filename}")