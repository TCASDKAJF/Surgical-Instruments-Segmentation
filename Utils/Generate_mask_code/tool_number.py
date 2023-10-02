import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def count_tools(df, testing_dir):
    tool_counts = {"background": 0}

    for root, dirs, files in os.walk(testing_dir):
        for file in files:
            if file.endswith(".jpg"):
                split_root = root.split("\\")
                video, clip, step = split_root[-1].split("_")[0:3]
                video = int(video[5:])
                clip = int(clip[4:])
                step = int(step[4:])
                sec = int(file.split("_")[-1].split(".")[0])

                row = df.loc[(df['video'] == video) & (df['clip'] == clip) & (df['step'] == step) & (df['sec'] == sec)]

                if len(row) == 0:
                    tool_counts["background"] += 1
                    continue

                tool_key = row['tool_key'].values[0]

                if tool_key == 18:
                    tool_counts["background"] += 1
                elif tool_key in [2, 8]:
                    tool_counts["kerrisons"] = tool_counts.get("kerrisons", 0) + 1
                else:
                    tool = row['tool'].values[0]
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return tool_counts

def plot_counts(counts):
    counts_df = pd.DataFrame(list(counts.items()), columns=["Tool", "Count"])
    counts_df = counts_df.sort_values("Count", ascending=False)

    plt.figure(figsize=(15, 10))
    sns.barplot(data=counts_df, x="Tool", y="Count")
    plt.title("Tool Counts in the Testing Dataset")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.pie(counts_df["Count"], labels=counts_df.index, autopct='%1.1f%%')
    plt.title("Tool Proportion in the Testing Dataset")
    plt.show()

    print(counts_df)

df = pd.read_csv("G:/Final_Data/segmentation/points_clean_none_removed.csv")
testing_dir = "G:/Final_Data/segmentation/testing"
testing_counts = count_tools(df, testing_dir)
plot_counts(testing_counts)
