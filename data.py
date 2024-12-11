import numpy as np
file_path = "./data/d00.dat"  # 替换为你的文件路径

def get_data(file_path):
    # 用来存储读取的数据
    data = []
    with open(file_path, "r") as file:
        # 按行读取文件
        for line in file:
            # 使用 split() 方法以空格分隔每一行数据
            # 默认情况下，split() 会处理多个空格，按空格进行拆分
            line_data = line.split()  # 返回一个字符串列表
            # 可以根据需要将数据转换为数字类型
            line_data = [float(i) for i in line_data]  # 假设数据是数字类型
            data.append(line_data)
    return np.array(data)

data_train = get_data("C:/Users/Soever/Downloads/d00.dat")
data_test  = get_data("./data/d00_te.dat")
print(data_train.shape)
print(data_test.shape)