import pandas as pd

df = pd.read_csv("2_f.csv")

# df_sorted = df.sort_values(by="Computer Vision & Image Processing", ascending=False)
df_sorted = df.sort_values(by="Artificial Intelligence & Machine Learning", ascending=False)

df_sorted.to_csv("ranked_2_f.csv", index=False)

print(df_sorted.head())
