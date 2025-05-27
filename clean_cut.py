import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# 数据读取（修复转义问题）
train = pd.read_csv('used_car_train_20200313.csv', sep=r'\s+', engine='python')
test = pd.read_csv('used_car_testB_20200421.csv', sep=r'\s+', engine='python')

# 数据预处理函数
def preprocess(df):
    # 统一缺失值表示
    df.replace('-', np.nan, inplace=True)

    # 日期处理
    for col in ['regDate', 'creatDate']:
        df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')

    # 时间特征工程
    df['regYear'] = df['regDate'].dt.year
    df['regMonth'] = df['regDate'].dt.month
    df['creatYear'] = df['creatDate'].dt.year
    df['reg_to_creat_days'] = (df['creatDate'] - df['regDate']).dt.days

    # 处理异常值
    df['power'] = pd.to_numeric(df['power'], errors='coerce').clip(0, 600)
    return df

# 应用预处理
train = preprocess(train)
test = preprocess(test)

# 定义特征和标签
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'price']
features = [col for col in train.columns if col not in drop_cols]
target = 'price'

# 缺失值处理
numeric_cols = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]
categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox',
                    'notRepairedDamage', 'regionCode', 'seller', 'offerType']

for df in [train, test]:
    # 数值型填充
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 类别型填充
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype('category')

# 对类别型特征进行编码
for col in categorical_cols:
    train[col] = train[col].cat.codes
    test[col] = test[col].cat.codes

X_train = train[features]
y_train = train[target]
X_test = test[features]

# 使用 LightGBM 模型
lgb_model = lgb.LGBMRegressor(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 40, 50]
}

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 预测
test_preds = best_model.predict(X_test)

# 生成提交文件
submission = pd.DataFrame({'SaleID': test['SaleID'], 'price': test_preds})
submission['price'] = submission['price'].round().astype(int)
submission.to_csv('submission.csv', index=False)