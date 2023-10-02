import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义函数以统计各个数据集中的视频数量
def count_videos(dir_path):
    video_counts = {}
    for dir_name in os.listdir(dir_path):
        if dir_name.startswith("video"):
            number = dir_name.split("_")[0][5:]
            video_counts[number] = video_counts.get(number, 0) + 1
    return video_counts

# 计算各个数据集中的视频数量
training_counts = count_videos("G:/Final_Data/segmentation/training")
testing_counts = count_videos("G:/Final_Data/segmentation/testing")
validation_counts = count_videos("G:/Final_Data/segmentation/validation")

# 生成数据框
df_train = pd.DataFrame(list(training_counts.items()), columns=['Video Number', 'Count'])
df_train['Set'] = 'Training'
df_test = pd.DataFrame(list(testing_counts.items()), columns=['Video Number', 'Count'])
df_test['Set'] = 'Testing'
df_val = pd.DataFrame(list(validation_counts.items()), columns=['Video Number', 'Count'])
df_val['Set'] = 'Validation'

df_all = pd.concat([df_train, df_test, df_val])

# 生成柱状图
plt.figure(figsize=(15,8))
sns.barplot(x='Video Number', y='Count', hue='Set', data=df_all)
plt.title('Video Counts in Each Dataset')
plt.show()
