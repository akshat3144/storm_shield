import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("/kaggle/input/noaa-powout-prism-0-1-is-storm-lag/noaapowoutprism_01_Is_Storm_Lag (1).csv")  # Update with your filename

# Step 1: Split into 80% (train+test) and 20% (holdout)
train_test, holdout = train_test_split(df, test_size=0.2, random_state=42)

# Step 2: Split train_test into 70% train and 10% test (out of total)
# Since train_test is 80%, we calculate relative proportions
relative_test_size = 0.1 / 0.8  # = 0.125 of train_test
train, test = train_test_split(train_test, test_size=relative_test_size, random_state=42)

# Step 3: Save each set to CSV
train.to_csv("/kaggle/working/train.csv", index=False)
test.to_csv("/kaggle/working/test.csv", index=False)
holdout.to_csv("/kaggle/working/holdout.csv", index=False)