import pandas as pd

df = pd.read_csv("data/Dengue_diseases_dataset_modified (1).csv")

print(df["gender"].value_counts(dropna=False))
print("\nUnique values:")
print(df["gender"].unique())