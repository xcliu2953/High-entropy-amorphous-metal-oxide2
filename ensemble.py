import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
import matplotlib.pyplot as plt

import shap

writer = pd.ExcelWriter('output3.xlsx', engine='openpyxl')

# 使用pandas读取CSV文件并创建DataFrame
df = pd.DataFrame(pd.read_csv('20230510_1.csv'))

X = np.array(df.drop(['C2', 'H2', 'O2','N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C'], axis=1))

param = ['C2', 'H2', 'O2', 'N2', 'S2', 'Solid Yield', 'VM', 'FC', 'Ash', 'TP', 'A-P', 'H/C', 'O/C']

shap_values_list = []  # 用于存储每个参数的Shapley值

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
    # 绘制散点图
    plt.scatter(y, y_pred)
    plt.plot([0.1, np.max(y)], [0.1, np.max(y)], color='red', linestyle='--', label='Line with slope 1')
    plt.xlabel('True Value')
    plt.ylabel('Predict Value')
    plt.title(param[i])
    plt.xlim(0.1, np.max(y))
    plt.ylim(0.1, np.max(y))
    plt.show()
    # 计算Shapley值
    explainer = shap.Explainer(rfr)
    shap_values = explainer(X)

    # 绘制每个输入变量对于输出变量的Shapley值图
    shap.summary_plot(shap_values, X, feature_names=df.columns[:-12], show=False)
    plt.title(f'{param[i]} - Shapley Values')
    param[i]=param[i].replace('/', '')
    plt.savefig(f'{param[i]}.png')
    plt.show()
    df_variance = pd.DataFrame({'原数据': y,
                                '预测值': y_pred,})
    sheet_name = param[i].replace('/', '')
    df_variance.to_excel(writer, sheet_name=sheet_name, index=False, startcol=0)
    # 创建DataFrame，并将其保存到工作表中
    df_shap = pd.DataFrame(shap_values.values, columns=df.columns[:-13])
    df_shap.to_excel(writer, sheet_name=sheet_name, index=False, startcol=2)
# 保存Excel文件
writer.save()


