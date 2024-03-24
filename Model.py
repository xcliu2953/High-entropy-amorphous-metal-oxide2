import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn import  metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from pygam import LinearGAM
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

writer = pd.ExcelWriter('output2.xlsx', engine='openpyxl')

# 使用pandas读取CSV文件并创建DataFrame
df = pd.DataFrame(pd.read_csv('20230510_1.csv'))

X = np.array(df.drop(['C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))

param = ['C2', 'H2', 'O2', 'N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C']

rfr_mae = []
rfr_mse = []
rfr_rmse = []
rfr_r2 = []
for i in range(0,len(param)):
    y = np.array(df[param[i]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    ensemble.RandomForestClassifier (n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                            min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                             class_weight=None, random_state=None, bootstrap=True, oob_score=False,
                                             n_jobs=None, verbose=0, warm_start=False)
    rfr = RandomForestRegressor(random_state=0)
    rfr = rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X)
    # 评价模型性能
    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, y_pred)

    rfr_mae.append(mae)
    rfr_rmse.append(rmse)
    rfr_mse.append(mse)
    rfr_r2.append(r2)
# 计算最小值和最大值
min_mae = min(rfr_mae)
max_mae = max(rfr_mae)
min_mse = min(rfr_mse)
max_mse = max(rfr_mse)
min_rmse = min(rfr_rmse)
max_rmse = max(rfr_rmse)
min_r2 = min(rfr_r2)
max_r2 = max(rfr_r2)
# 归一化 rfr_mae
rfr_mae = [(value - min_mae) / (max_mae - min_mae) for value in rfr_mae]
# 归一化 rfr_mse
rfr_mse = [(value - min_mse) / (max_mse - min_mse) for value in rfr_mse]
# 归一化 rfr_rmse
rfr_rmse = [(value - min_rmse) / (max_rmse - min_rmse) for value in rfr_rmse]
# 归一化 rfr_r2
rfr_r2 = [(value - min_r2) / (max_r2 - min_r2) for value in rfr_r2]
# 找到每个列表的最大值和最小值
min_rfr_mae = min(rfr_mae)
max_rfr_mae = max(rfr_mae)

min_rfr_mse = min(rfr_mse)
max_rfr_mse = max(rfr_mse)

min_rfr_rmse = min(rfr_rmse)
max_rfr_rmse = max(rfr_rmse)

min_rfr_r2 = min(rfr_r2)
max_rfr_r2 = max(rfr_r2)

# 创建新的列表，分别排除最大值和最小值
new_rfr_mae = [x for x in rfr_mae if x != min_rfr_mae and x != max_rfr_mae]
new_rfr_mse = [x for x in rfr_mse if x != min_rfr_mse and x != max_rfr_mse]
new_rfr_rmse = [x for x in rfr_rmse if x != min_rfr_rmse and x != max_rfr_rmse]
new_rfr_r2 = [x for x in rfr_r2 if x != min_rfr_r2 and x != max_rfr_r2]

# 将新列表赋值回原始变量
rfr_mae = new_rfr_mae
rfr_mse = new_rfr_mse
rfr_rmse = new_rfr_rmse
rfr_r2 = new_rfr_r2

df_variance = pd.DataFrame({'mae': rfr_mae,
                            'mse': rfr_mse,
                            'rmse': rfr_rmse,
                            'r2': rfr_r2,})

df_variance.to_excel(writer, sheet_name='rfr', index=False, startcol=0)

# 输出结果
print('RFR:')
print("rfr_mae:", rfr_mae)
print("rfr_mse:", rfr_mse)
print("rfr_rmse:", rfr_rmse)
print("rfr_r2:", rfr_r2)



nnr_mae = []
nnr_mse = []
nnr_rmse = []
nnr_r2 = []
for i in range(0,len(param)):
    y = np.array(df[param[i]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    nnr = KNeighborsRegressor(n_neighbors=5)  # 使用5个最近邻进行回归
    nnr.fit(X_train, y_train)
    y_pred = nnr.predict(X)
    # 评价模型性能
    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, y_pred)
    nnr_mae.append(mae)
    nnr_rmse.append(rmse)
    nnr_mse.append(mse)
    nnr_r2.append(r2)
# 计算最小值和最大值
min_mae = min(nnr_mae)
max_mae = max(nnr_mae)
min_mse = min(nnr_mse)
max_mse = max(nnr_mse)
min_rmse = min(nnr_rmse)
max_rmse = max(nnr_rmse)
min_r2 = min(nnr_r2)
max_r2 = max(nnr_r2)
# 归一化 nnr_mae
nnr_mae = [(value - min_mae) / (max_mae - min_mae) for value in nnr_mae]
# 归一化 nnr_mse
nnr_mse = [(value - min_mse) / (max_mse - min_mse) for value in nnr_mse]
# 归一化 nnr_rmse
nnr_rmse = [(value - min_rmse) / (max_rmse - min_rmse) for value in nnr_rmse]
# 归一化 nnr_r2
nnr_r2 = [(value - min_r2) / (max_r2 - min_r2) for value in nnr_r2]
# 找到每个列表的最大值和最小值
min_nnr_mae = min(nnr_mae)
max_nnr_mae = max(nnr_mae)

min_nnr_mse = min(nnr_mse)
max_nnr_mse = max(nnr_mse)

min_nnr_rmse = min(nnr_rmse)
max_nnr_rmse = max(nnr_rmse)

min_nnr_r2 = min(nnr_r2)
max_nnr_r2 = max(nnr_r2)

# 创建新的列表，分别排除最大值和最小值
new_nnr_mae = [x for x in nnr_mae if x != min_nnr_mae and x != max_nnr_mae]
new_nnr_mse = [x for x in nnr_mse if x != min_nnr_mse and x != max_nnr_mse]
new_nnr_rmse = [x for x in nnr_rmse if x != min_nnr_rmse and x != max_nnr_rmse]
new_nnr_r2 = [x for x in nnr_r2 if x != min_nnr_r2 and x != max_nnr_r2]

# 将新列表赋值回原始变量
nnr_mae = new_nnr_mae
nnr_mse = new_nnr_mse
nnr_rmse = new_nnr_rmse
nnr_r2 = new_nnr_r2

df_variance = pd.DataFrame({'mae': nnr_mae,
                            'mse': nnr_mse,
                            'rmse': nnr_rmse,
                            'r2': nnr_r2,})

df_variance.to_excel(writer, sheet_name='nnr', index=False, startcol=0)

# 输出结果
print('NNR:')
print("nnr_mae:", nnr_mae)
print("nnr_mse:", nnr_mse)
print("nnr_rmse:", nnr_rmse)
print("nnr_r2:", nnr_r2)


gam_mae = []
gam_mse = []
gam_rmse = []
gam_r2 = []
for i in range(0,len(param)):
    y = np.array(df[param[i]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    gam = LinearGAM()  # 创建广义可加模型
    gam.fit(X_train, y_train)
    y_pred = gam.predict(X)
    # 评价模型性能
    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, y_pred)
    gam_mae.append(mae)
    gam_rmse.append(rmse)
    gam_mse.append(mse)
    gam_r2.append(r2)
# 计算最小值和最大值
min_mae = min(gam_mae)
max_mae = max(gam_mae)
min_mse = min(gam_mse)
max_mse = max(gam_mse)
min_rmse = min(gam_rmse)
max_rmse = max(gam_rmse)
min_r2 = min(gam_r2)
max_r2 = max(gam_r2)
# 归一化 gam_mae
gam_mae = [(value - min_mae) / (max_mae - min_mae) for value in gam_mae]
# 归一化 gam_mse
gam_mse = [(value - min_mse) / (max_mse - min_mse) for value in gam_mse]
# 归一化 gam_rmse
gam_rmse = [(value - min_rmse) / (max_rmse - min_rmse) for value in gam_rmse]
# 归一化 gam_r2
gam_r2 = [(value - min_r2) / (max_r2 - min_r2) for value in gam_r2]
# 找到每个列表的最大值和最小值
min_mae = min(gam_mae)
max_mae = max(gam_mae)

min_mse = min(gam_mse)
max_mse = max(gam_mse)

min_rmse = min(gam_rmse)
max_rmse = max(gam_rmse)

min_r2 = min(gam_r2)
max_r2 = max(gam_r2)

# 创建新的列表，分别排除最大值和最小值
new_gam_mae = [x for x in gam_mae if x != min_mae and x != max_mae]
new_gam_mse = [x for x in gam_mse if x != min_mse and x != max_mse]
new_gam_rmse = [x for x in gam_rmse if x != min_rmse and x != max_rmse]
new_gam_r2 = [x for x in gam_r2 if x != min_r2 and x != max_r2]

# 将新列表赋值回原始变量
gam_mae = new_gam_mae
gam_mse = new_gam_mse
gam_rmse = new_gam_rmse
gam_r2 = new_gam_r2

df_variance = pd.DataFrame({'mae': gam_mae,
                            'mse': gam_mse,
                            'rmse': gam_rmse,
                            'r2': gam_r2,})

df_variance.to_excel(writer, sheet_name='gam', index=False, startcol=0)
print('GAM:')
print('MAE:', gam_mae)
print('MSE:', gam_mse)
print('RMSE:', gam_rmse)
print('R2:', gam_r2)

svr_mae = []
svr_mse = []
svr_rmse = []
svr_r2 = []
for i in range(0,len(param)):
    y = np.array(df[param[i]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    svr = SVR()  # 创建支持向量回归器
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X)
    # 评价模型性能
    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, y_pred)
    svr_mae.append(mae)
    svr_rmse.append(rmse)
    svr_mse.append(mse)
    svr_r2.append(r2)
# 计算最小值和最大值
min_mae = min(svr_mae)
max_mae = max(svr_mae)
min_mse = min(svr_mse)
max_mse = max(svr_mse)
min_rmse = min(svr_rmse)
max_rmse = max(svr_rmse)
min_r2 = min(svr_r2)
max_r2 = max(svr_r2)
# 归一化 svr_mae
svr_mae = [(value - min_mae) / (max_mae - min_mae) for value in svr_mae]
# 归一化 svr_mse
svr_mse = [(value - min_mse) / (max_mse - min_mse) for value in svr_mse]
# 归一化 svr_rmse
svr_rmse = [(value - min_rmse) / (max_rmse - min_rmse) for value in svr_rmse]
# 归一化 svr_r2
svr_r2 = [(value - min_r2) / (max_r2 - min_r2) for value in svr_r2]
# 找到每个列表的最大值和最小值
min_svr_mae = min(svr_mae)
max_svr_mae = max(svr_mae)

min_svr_mse = min(svr_mse)
max_svr_mse = max(svr_mse)

min_svr_rmse = min(svr_rmse)
max_svr_rmse = max(svr_rmse)

min_svr_r2 = min(svr_r2)
max_svr_r2 = max(svr_r2)

# 创建新的列表，分别排除最大值和最小值
new_svr_mae = [x for x in svr_mae if x != min_svr_mae and x != max_svr_mae]
new_svr_mse = [x for x in svr_mse if x != min_svr_mse and x != max_svr_mse]
new_svr_rmse = [x for x in svr_rmse if x != min_svr_rmse and x != max_svr_rmse]
new_svr_r2 = [x for x in svr_r2 if x != min_svr_r2 and x != max_svr_r2]

# 将新列表赋值回原始变量
svr_mae = new_svr_mae
svr_mse = new_svr_mse
svr_rmse = new_svr_rmse
svr_r2 = new_svr_r2

df_variance = pd.DataFrame({'mae': svr_mae,
                            'mse': svr_mse,
                            'rmse': svr_rmse,
                            'r2': svr_r2,})

df_variance.to_excel(writer, sheet_name='svr', index=False, startcol=0)
# 输出结果
print('SVR:')
print("svr_mae:", svr_mae)
print("svr_mse:", svr_mse)
print("svr_rmse:", svr_rmse)
print("svr_r2:", svr_r2)


gpr_mae = []
gpr_mse = []
gpr_rmse = []
gpr_r2 = []
for i in range(0,len(param)):
    y = np.array(df[param[i]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    gpr = GaussianProcessRegressor()  # 创建高斯过程回归器
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X)
    # 评价模型性能
    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, y_pred)
    gpr_mae.append(mae)
    gpr_rmse.append(rmse)
    gpr_mse.append(mse)
    gpr_r2.append(r2)
# 计算最小值和最大值
min_mae = min(gpr_mae)
max_mae = max(gpr_mae)
min_mse = min(gpr_mse)
max_mse = max(gpr_mse)
min_rmse = min(gpr_rmse)
max_rmse = max(gpr_rmse)
min_r2 = min(gpr_r2)
max_r2 = max(gpr_r2)
# 归一化 gpr_mae
gpr_mae = [(value - min_mae) / (max_mae - min_mae) for value in gpr_mae]
# 归一化 gpr_mse
gpr_mse = [(value - min_mse) / (max_mse - min_mse) for value in gpr_mse]
# 归一化 gpr_rmse
gpr_rmse = [(value - min_rmse) / (max_rmse - min_rmse) for value in gpr_rmse]
# 归一化 gpr_r2
gpr_r2 = [(value - min_r2) / (max_r2 - min_r2) for value in gpr_r2]
# 找到每个列表的最大值和最小值
min_gpr_mae = min(gpr_mae)
max_gpr_mae = max(gpr_mae)

min_gpr_mse = min(gpr_mse)
max_gpr_mse = max(gpr_mse)

min_gpr_rmse = min(gpr_rmse)
max_gpr_rmse = max(gpr_rmse)

min_gpr_r2 = min(gpr_r2)
max_gpr_r2 = max(gpr_r2)

# 创建新的列表，分别排除最大值和最小值
new_gpr_mae = [x for x in gpr_mae if x != min_gpr_mae and x != max_gpr_mae]
new_gpr_mse = [x for x in gpr_mse if x != min_gpr_mse and x != max_gpr_mse]
new_gpr_rmse = [x for x in gpr_rmse if x != min_gpr_rmse and x != max_gpr_rmse]
new_gpr_r2 = [x for x in gpr_r2 if x != min_gpr_r2 and x != max_gpr_r2]

# 将新列表赋值回原始变量
gpr_mae = new_gpr_mae
gpr_mse = new_gpr_mse
gpr_rmse = new_gpr_rmse
gpr_r2 = new_gpr_r2

df_variance = pd.DataFrame({'mae': gpr_mae,
                            'mse': gpr_mse,
                            'rmse': gpr_rmse,
                            'r2': gpr_r2,})

df_variance.to_excel(writer, sheet_name='gpr', index=False, startcol=0)
# 输出结果
print('GPR:')
print("gpr_mae:", gpr_mae)
print("gpr_mse:", gpr_mse)
print("gpr_rmse:", gpr_rmse)
print("gpr_r2:", gpr_r2)

# 五个列表数据
data = [rfr_mae, nnr_mae, gam_mae, svr_mae, gpr_mae]
labels = ['RFR', 'NNR', 'GAM', 'SVR', 'GPR']
# 绘制小提琴图
sns.violinplot(data=data)
# 设置横坐标刻度和标签
plt.xticks(range(len(data)), labels)
# 添加标题和标签
plt.title("Violin Plot of MAE")
plt.xlabel("Models")
plt.ylabel("MAE")
# 显示图形
plt.show()

# 五个列表数据
data = [rfr_mse, nnr_mse, gam_mse, svr_mse, gpr_mse]
labels = ['RFR', 'NNR', 'GAM', 'SVR', 'GPR']
# 绘制小提琴图
sns.violinplot(data=data)
# 设置横坐标刻度和标签
plt.xticks(range(len(data)), labels)
# 添加标题和标签
plt.title("Violin Plot of MSE")
plt.xlabel("Models")
plt.ylabel("MSE")
# 显示图形
plt.show()

# 五个列表数据
data = [rfr_rmse, nnr_rmse, gam_rmse, svr_rmse, gpr_rmse]
labels = ['RFR', 'NNR', 'GAM', 'SVR', 'GPR']
# 绘制小提琴图
sns.violinplot(data=data)
# 设置横坐标刻度和标签
plt.xticks(range(len(data)), labels)
# 添加标题和标签
plt.title("Violin Plot of RMSE")
plt.xlabel("Models")
plt.ylabel("RMSE")
# 显示图形
plt.show()

# 五个列表数据
data = [rfr_r2, nnr_r2, gam_r2, svr_r2, gpr_r2]
labels = ['RFR', 'NNR', 'GAM', 'SVR', 'GPR']
# 绘制小提琴图
sns.violinplot(data=data)
# 设置横坐标刻度和标签
plt.xticks(range(len(data)), labels)
# 添加标题和标签
plt.title("Violin Plot of R2 Score")
plt.xlabel("Models")
plt.ylabel("R2 Score")
# 显示图形
plt.show()
writer.save()