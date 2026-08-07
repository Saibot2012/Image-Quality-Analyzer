import pandas as pd
df = pd.read_csv("eye_dataset.csv")

print(df[(df["class"] == "closed") & (df["right_ear"] > 0.20)][["filename","left_ear","right_ear"]])
print(df[(df["class"] == "right_closed") & (df["left_ear"] < 0.12)][["filename","left_ear","right_ear"]])