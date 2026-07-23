import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("eye_dataset.csv")

open_df = df[df["label"] == 0]
closed_df = df[df["label"] == 1]

plt.figure(figsize=(8,5))

plt.hist(
    open_df["avg_ear"],
    bins=15,
    alpha=0.6,
    label="Open",
)

plt.hist(
    closed_df["avg_ear"],
    bins=15,
    alpha=0.6,
    label="Closed",
)

plt.xlabel("Average EAR")
plt.ylabel("Frequency")
plt.title("Open vs Closed Eye EAR Distribution")
plt.legend()

plt.grid(alpha=0.3)

plt.show()

print("\nOpen Eyes")
print(open_df["avg_ear"].describe())

print("\nClosed Eyes")
print(closed_df["avg_ear"].describe())