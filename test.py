import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
def data():
    data = list(range(1, 51))
    random.shuffle(data)
    return data

def bubbleSort(data):
    for i in range(len(data)):
        for j in range(i):
            if data[i]>data[j]:
                data[i],data[j]=data[j],data[i]
                plt.bar(range(1, 51), Data)
                plt.show()
                time.sleep(0.1)
Data=data()
print(Data)

# 创建一些示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 创建图像和坐标轴对象
# fig, ax = plt.subplots()

# # 绘制数据
# ax.plot(x, y)

# # 创建边界框
# bbox = patches.Rectangle((2, 3), 2, 4, linewidth=2, edgecolor='r', facecolor='none')

# # 添加边界框到坐标轴上
# ax.add_patch(bbox)

# # 设置坐标轴标签和标题
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('边界框示例')

# # 显示图像

bubbleSort(Data)