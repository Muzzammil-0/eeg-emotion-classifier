import pandas as pd

df1 = pd.read_csv('emotions.csv')
df2 = pd.read_csv('emotions_27.csv')

print("Bird dataset:", df1.shape)
print("EEGEmotions-27:", df2.shape)
print("\nBird labels:\n", df1['label'].value_counts())
print("\nEEG27 labels:\n", df2['label'].value_counts())

combined = pd.concat([df1, df2], ignore_index=True)
combined.to_csv('emotions_combined.csv', index=False)

print("\nCombined:", combined.shape)
print("\nCombined labels:\n", combined['label'].value_counts())