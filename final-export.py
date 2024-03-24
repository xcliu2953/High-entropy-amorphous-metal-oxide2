import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
import matplotlib.pyplot as plt
# import shap
from sklearn import  metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# writer = pd.ExcelWriter('output1.xlsx', engine='openpyxl')


df = pd.DataFrame(pd.read_csv('matched_letter.csv'))

X = np.array(df.drop(['configuration', 'Energy', 'output','Mn', 'Co', 'Ni', 'Ru', 'Ir'], axis=1))

y = np.array(df[['output']])
correlation_matrix, p_values = spearmanr(X, y)

fig, ax = plt.subplots(figsize=(40, 32))

sns.heatmap(correlation_matrix, annot=False, cmap="YlGnBu", ax=ax)
# 调整x轴和y轴刻度标签的字体大小
ax.tick_params(axis='x', labelsize=12)  # 设置x轴刻度标签的字体大小
ax.tick_params(axis='y', labelsize=12)  # 设置y轴刻度标签的字体大小
plt.savefig("heatmap.png")
# Show the plot
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                class_weight=None, random_state=None, bootstrap=True, oob_score=False,
                                n_jobs=None, verbose=0, warm_start=False)
rfr = RandomForestRegressor(random_state=0)
rfr = rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

# 评价模型性能
rfr_mae = metrics.mean_absolute_error(y_test, y_pred)
rfr_mse = metrics.mean_squared_error(y_test, y_pred)
rfr_rmse = np.sqrt(rfr_mse)
rfr_r2 = metrics.r2_score(y_test, y_pred)
print('RFR:')
print("rfr_mae:", rfr_mae)
print("rfr_mse:", rfr_mse)
print("rfr_rmse:", rfr_rmse)
print("rfr_r2:", rfr_r2)
# 保存预测结果到DataFrame
results_rfr = pd.DataFrame({'Actual Values': y_test.flatten(), 'Predicted Values': y_pred.flatten()})

# 保存结果到Excel
results_rfr.to_excel("RFR_results.xlsx", index=False)

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', alpha=0.5)
# 添加标签和标题
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('RFR:Actual vs. Predicted Values')
# 绘制一条对角线，表示完美预测的情况
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
# 显示图形
plt.grid(True)
plt.savefig("RFR.png")
plt.show()

# NNR模型：
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
nnr = KNeighborsRegressor(n_neighbors=5)  # 使用5个最近邻进行回归
nnr.fit(X_train, y_train)
y_pred = nnr.predict(X_test)
# 评价模型性能
nnr_mae = metrics.mean_absolute_error(y_test, y_pred)
nnr_mse = metrics.mean_squared_error(y_test, y_pred)
nnr_rmse = np.sqrt(nnr_mse)
nnr_r2 = metrics.r2_score(y_test, y_pred)
print('NNR:')
print("nnr_mae:", nnr_mae)
print("nnr_mse:", nnr_mse)
print("nnr_rmse:", nnr_rmse)
print("nnr_r2:", nnr_r2)
# 保存预测结果到DataFrame
results_nnr = pd.DataFrame({'Actual Values': y_test.flatten(), 'Predicted Values': y_pred.flatten()})

# 保存结果到Excel
results_nnr.to_excel("NNR_results.xlsx", index=False)


# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', alpha=0.5)
# 添加标签和标题
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('NNR:Actual vs. Predicted Values')
# 绘制一条对角线，表示完美预测的情况
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
# 显示图形
plt.grid(True)
plt.savefig("NNR.png")
plt.show()

# SVR模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
svr = SVR()  # 创建支持向量回归器
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
# 评价模型性能
svr_mae = metrics.mean_absolute_error(y_test, y_pred)
svr_mse = metrics.mean_squared_error(y_test, y_pred)
svr_rmse = np.sqrt(svr_mse)
svr_r2 = metrics.r2_score(y_test, y_pred)
print('SVR:')
print("svr_mae:", svr_mae)
print("svr_mse:", svr_mse)
print("svr_rmse:", svr_rmse)
print("svr_r2:", svr_r2)
# 保存预测结果到DataFrame
results_svr = pd.DataFrame({'Actual Values': y_test.flatten(), 'Predicted Values': y_pred.flatten()})

# 保存结果到Excel
results_svr.to_excel("SVR_results.xlsx", index=False)


# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', alpha=0.5)
# 添加标签和标题
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR:Actual vs. Predicted Values')
# 绘制一条对角线，表示完美预测的情况
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
# 显示图形
plt.grid(True)
plt.savefig("SVR.png")
plt.show()

# # 创建一个 explainer 对象，用于解释模型
# explainer = shap.Explainer(rfr, X_train)
# # 计算 Shapley 值
# shap_values = explainer(X_test)
# # 绘制 Shapley 摘要图
# # 使用手动指定的特征名称
# shap.summary_plot(shap_values, X_test, feature_names=['Mn1', 'Mn2', 'Mn3','Co1','Co2','Co3','Co4','Ni1', 'Ni2', 'Ni3','Ru1', 'Ru2', 'Ru3','Ir1', 'Ir2', 'Ir3',], show=False)
# # 显示图形
# plt.savefig("shap.png")
# plt.show()

