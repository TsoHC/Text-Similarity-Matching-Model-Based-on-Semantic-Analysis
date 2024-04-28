from datasets import load_dataset

# 加载SICK数据集
dataset = load_dataset("sick.py")

# 查看数据集的信息
print(dataset)

# 访问训练集、验证集和测试集
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# 查看训练集的第一个样本
print(train_dataset[0])