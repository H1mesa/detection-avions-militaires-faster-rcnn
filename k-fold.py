import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

img_dir = "vignettes_avions"
data = []
for filename in os.listdir(img_dir):
    if filename.endswith(".jpg"):
        label = filename.split('_')[0]
        data.append({'filename': filename, 'label': label})

df = pd.DataFrame(data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Stratified K-Fold
df['fold'] = -1
for fold_id, (_, val_idx) in enumerate(skf.split(df['filename'], df['label'])):
    df.loc[val_idx, 'fold'] = fold_id

df.to_csv("dataset_kfold.csv", index=False)