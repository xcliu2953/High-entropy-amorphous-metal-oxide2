import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

writer = pd.ExcelWriter('output1.xlsx', engine='openpyxl')


df = pd.DataFrame(pd.read_csv('matched_letter.csv'))

X = np.array(df.drop(['C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))

y = np.array(df[['C2', 'H2', 'O2', 'N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C']])

correlation_matrix, p_values = spearmanr(X, y)

fig, ax = plt.subplots(figsize=(40, 32))

sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", ax=ax)
plt.savefig("heatmap.png")
# Show the plot

plt.show()

# 设置相关性矩阵的横纵坐标为CSV文件对应的列名
plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45, ha='right')
plt.yticks(np.arange(len(df.columns)), df.columns, rotation=45, ha='right')

plt.tight_layout()




scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 进行主成分分析
pca = PCA()
df_pca = pca.fit_transform(X_scaled)


explained_variance = pca.explained_variance_
cumulative_variance = np.cumsum(explained_variance)

df_variance = pd.DataFrame({'Explained Variance': explained_variance,
                            'Cumulative Variance': cumulative_variance})

df_variance.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0)


plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(explained_variance) + 1), explained_variance, label='Individual Variance')
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='r', label='Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.title('PCA - Individual Variance and Cumulative Variance')
plt.legend()
plt.xticks(np.arange(1, len(explained_variance) + 1), np.arange(1, len(explained_variance) + 1), rotation=45, ha='right')
plt.savefig('pca_variance.png')
plt.show()

parameters = df.drop(['residence time','reaction temperature', 'C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1).columns


X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)

df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='Solid Yield', index=False, startcol=0)


plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on Solid Yield')
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()


X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='VM', index=False, startcol=0)

plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on VM')
# 添加横线和竖线
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

# 绘制输入变量与FC之间的PCA相关性
X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='FC', index=False, startcol=0)

plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on FC')
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

# 绘制输入变量与Ash之间的PCA相关性
X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='Ash', index=False, startcol=0)

plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on Ash')
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)


for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'A-P', 'H/C', 'O/C'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='TP', index=False, startcol=0)

# 绘制散点图
plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on TP')
# 添加横线和竖线
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

# 绘制输入变量与A-P之间的PCA相关性
X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'H/C', 'O/C'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='A-P', index=False, startcol=0)

# 绘制散点图
plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on A-P')
# 添加横线和竖线
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

# 标记参数名称
for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

# 绘制输入变量与H/C之间的PCA相关性
X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'O/C', 'A-P'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='H-C', index=False, startcol=0)

# 绘制散点图
plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on H/C')
# 添加横线和竖线
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

# 标记参数名称
for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

# 绘制输入变量与O/C之间的PCA相关性
X = np.array(df.drop(['residence time','reaction temperature','C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'H/C', 'A-P'], axis=1))
pca = PCA(n_components=2)  # 设置提取的主成分个数为2，即PC1和PC2
X_pca = pca.fit_transform(X)
df_variance = pd.DataFrame({'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1]})

df_variance.to_excel(writer, sheet_name='O-C', index=False, startcol=0)

# 绘制散点图
plt.figure(figsize=(20, 12))  # 设置图形的大小
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Impact of Input Variables on O/C')
# 添加横线和竖线
plt.axhline(np.mean(X_pca[:, 1]), color='black', linestyle='--')
plt.axvline(np.mean(X_pca[:, 0]), color='black', linestyle='--')
plt.xlim(-200, 200)

# 标记参数名称
for i in range(len(parameters)):
    param_indices = np.where(parameters == parameters[i])[0]  # 获取与当前参数匹配的数据点索引
    param_points = X_pca[param_indices]  # 选择相应的数据点坐标
    x_mean = np.mean(param_points[:, 0])
    y_mean = np.mean(param_points[:, 1])
    offset_x = np.random.uniform(-40, 40)
    offset_y = np.random.uniform(-40, 40)
    plt.text(x_mean + offset_x, y_mean + offset_y, parameters[i], fontsize=12, fontweight='bold', color='red', ha='center', va='center')

plt.show()

writer.save()