import matplotlib.pyplot as plt
import numpy as np

# 定义数据
x = np.random.rand(10)  # 取出10个随机数
y = x + x ** 2 - 10  # 用自定义关系确定y的值

# 绘图
# 1. 确定画布
plt.figure(figsize=(5, 4))  # figsize:确定画布大小

plt.xlim(0, 1)
plt.ylim(0.5, 2)

y=[1.978,1.675,1.653,1.262,1.316,1.092,1.216,0.690]
x=[0.774,0.618,0.610,0.517,0.461 , 0.342,0.675,0.206]
l=['Bilatera','MRF','AD','FCN','Zhang','Huang','Pre-training(Ours)','Fine-tuning(Ours)']
# for i in range(len(x)-1):
#     plt.scatter(x[i], y[i], s=None, c='b', label='function',marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,edgecolors=None,  data=None)


plt.scatter(x[0], y[0], marker='+', color='b',label=l[0])
plt.scatter(x[1], y[1], marker='x', color='g',label=l[1])
plt.scatter(x[2], y[2], marker='o', color='c',label=l[2])
plt.scatter(x[3], y[3], marker='s', color='m',label=l[3])
plt.scatter(x[4], y[4], marker='h', color='y',label=l[4])
plt.scatter(x[5], y[5], marker=',', color='b',label=l[5])
plt.scatter(x[6], y[6], marker='>', color='skyblue',label=l[6])
plt.scatter(x[-1], y[-1], marker='p', color='r',linewidths=4,label=l[-1])

# # 2. 绘图
# plt.scatter(x[1],  # 横坐标
#             y[1],  # 纵坐标
#             c='red',  # 点的颜色
#             label='function')  # 标签 即为点代表的意思
# 3.展示图形
plt.rcParams.update({'font.size': 8})
plt.legend(loc="lower right")
# plt.legend(loc="upper left")
# plt.legend()  # 显示图例
plt.xlabel('ME')
plt.ylabel('RMSE')
plt.show()  # 显示所绘图形
