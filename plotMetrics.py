import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lightning_logs/version_2/metrics.csv", sep = ",")

print(df.head())

print(np.argmin(df['val_loss']))
print(np.max(df['val_acc']))
print(df.iloc[np.argmax(df['val_acc']), 2])
#plt.plot(df.loc[~df['train_loss'].isna(), 'train_loss'].values, label = "Train Loss")
plt.figure()
plt.plot(df.loc[~df['val_acc'].isna(), 'val_acc'].values, label = "Val Acc")
plt.legend()
plt.savefig("loss_curve.png")