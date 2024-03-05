import pandas as pd
import random

def create_val_csv(input_csv_path, val_csv_path):
    # 读取原始CSV文件
    df = pd.read_csv(input_csv_path)

    # 计算13%的数据量
    val_percentage = 0.16
    val_size = int(len(df) * val_percentage)

    # 随机选择13%的数据
    val_data = df.sample(n=val_size, random_state=42)

    # 保存验证数据到val CSV文件
    val_data.to_csv(val_csv_path, index=False)

# 用法示例
input_csv_path = r'D:\a研究\MolLoG\datasets\drugbank\train.csv'  # 替换为你的原始CSV文件路径
val_csv_path = r'D:\a研究\MolLoG\datasets\drugbank\val.csv'       # 替换为你想要保存val CSV文件的路径

create_val_csv(input_csv_path, val_csv_path)