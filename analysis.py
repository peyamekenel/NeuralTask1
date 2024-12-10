import pandas as pd
import numpy as np

# Load the datasets
labeled = pd.read_csv('~/attachments/iBeacon_RSSI_Labeled.csv')
unlabeled = pd.read_csv('~/attachments/iBeacon_RSSI_Unlabeled.csv')

# Print basic information
print(f'Labeled data shape: {labeled.shape}')
print(f'Unlabeled data shape: {unlabeled.shape}')
print('\nUnique locations:', sorted(labeled['location'].unique()))

# Analyze RSSI value ranges
print('\nRSSI value ranges (excluding -200):')
beacon_cols = [col for col in labeled.columns if col.startswith('b300')]
for col in beacon_cols:
    values = labeled[col][labeled[col] != -200]
    if len(values) > 0:
        print(f'{col}: [{values.min()}, {values.max()}]')

# Additional statistics
print('\nPercentage of -200 values per beacon:')
for col in beacon_cols:
    pct = (labeled[col] == -200).mean() * 100
    print(f'{col}: {pct:.1f}%')
