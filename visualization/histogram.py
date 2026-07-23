import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("ml/eye_analysis_dataset.csv")

# Colours
colours = {
    "Eyes open": "blue",
    "Eyes closed": "red",
    "One eye closed": "orange",
    "Eyes unclear": "gray"
}

plt.figure(figsize=(8,6))

for status in df["predicted_status"].unique():

    subset = df[df["predicted_status"] == status]

    plt.scatter(
        subset["average_ear"],
        subset["ratio"],
        s=35,
        alpha=0.7,
        label=status,
        color=colours.get(status, "black")
    )

plt.xlabel("Average EAR")
plt.ylabel("EAR Ratio")
plt.title("Average EAR vs Ratio")
plt.grid(True)
plt.legend()

plt.show()
plt.savefig("ear_ratio_scatter.png", dpi=300)