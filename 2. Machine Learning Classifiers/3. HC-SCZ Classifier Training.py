import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from optuna.integration import SkoptSampler
import pandas as pd
import optuna
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score
import joblib
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV  # 用于 Platt Scaling 校准
import os

# -------------------------------
# 1. 加载数据
with open('./data/variables-huafen.pkl', 'rb') as file:
    X_resampled, y_resampled = pickle.load(file)

# 排除部分特征
exclusive = ['Dose Equivalent to Olanzapine', 'PANSS-N', 'PANSS-P', 'PANSS-GP', 'PANSS-T']
X = X_resampled[list(set(X_resampled.columns.tolist()) - set(exclusive))]
y = y_resampled.to_numpy()

# -------------------------------
# 2. 定义 Optuna 超参数优化目标函数
def optuna_opti(trial, x, y):
    params = {'random_state': 2025, 'probability': True}
    params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params['C'] = trial.suggest_float('C', 0.001, 0.1)
    params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None])
    clf = SVC(**params)
    
    clf.fit(x, y)
    train_acc = balanced_accuracy_score(y, clf.predict(x))
    return train_acc

# -------------------------------
# 3. 主流程设置
epoch = 100

inner_metrics_list = []  
out_metrics_list = []   
hold_metrics_list = []  
hold_params_list = []  

np.random.seed(42)
random_numbers = np.random.rand(100)
random_numbers = [int(x * 1000) for x in random_numbers]

global_best_item = {}
global_best_item_list = []

model_name = "SVM"  # 固定使用 SVM 模型

# 确保输出文件夹存在
for folder in ['./StandarScaler', './split_csv', './model_history_holdout', './optuna', 'metrics']:
    if not os.path.exists(folder):
        os.makedirs(folder)

for _ in tqdm(range(epoch)):
    best_auc = 0
    best_params = {}
    best_params_list = []
    scaler = StandardScaler()

    # 划分训练集和保留集（holdout）
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_numbers[_]
    )
    train_indices = X_train.index
    test_indices = X_holdout.index
    
    # 标准化：对训练集拟合，训练集和保留集都进行转换
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_train_scaled.index = train_indices
    X_holdout_scaled = pd.DataFrame(scaler.transform(X_holdout), columns=X_train.columns)
    X_holdout_scaled.index = test_indices

    # 保存标准化器和分割信息
    joblib.dump(scaler, f'./StandarScaler/{model_name}{_}.pkl')
    with pd.ExcelWriter(f'./split_csv/{model_name}_indices{_}.xlsx') as writer:
        temp1 = deepcopy(X_train_scaled)
        temp1['Y'] = y_train
        temp2 = deepcopy(X_holdout_scaled)
        temp2['Y'] = y_holdout
        X_resampled[exclusive].iloc[train_indices].to_excel(writer, sheet_name='Train Data')
        X_resampled[exclusive].iloc[test_indices].to_excel(writer, sheet_name='Test Data')
        temp1.to_excel(writer, sheet_name='Hold Train')
        temp2.to_excel(writer, sheet_name='Hold Test')

    # 将 DataFrame 转换为 numpy 数组以便后续处理
    X_train_arr = X_train_scaled.to_numpy()
    X_holdout_arr = X_holdout_scaled.to_numpy()
    
    # 定义外层和内层交叉验证
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 外层交叉验证
    for outer_train_idx, outer_test_idx in outer_cv.split(X_train_arr, y_train):
        X_train_outer, X_test_outer = X_train_arr[outer_train_idx], X_train_arr[outer_test_idx]
        y_train_outer, y_test_outer = y_train[outer_train_idx], y_train[outer_test_idx]

        # 内层交叉验证
        for inner_train_idx, inner_test_idx in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner, X_test_inner = X_train_arr[inner_train_idx], X_train_arr[inner_test_idx]
            y_train_inner, y_test_inner = y_train[inner_train_idx], y_train[inner_test_idx] 

            # 使用 Optuna 进行超参数搜索
            algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP', 'n_initial_points': 10, 'acq_func': 'EI'})
            study = optuna.create_study(sampler=algo, direction="maximize")
            study.optimize(lambda trial: optuna_opti(trial, X_train_inner, y_train_inner), 
                           n_trials=10, show_progress_bar=True)
            best_params_list.append(study.best_trial)
            best_trial = study.best_trial

            if best_trial.value > best_auc:
                best_auc = best_trial.value
                best_params = best_trial.params

            best_params['probability'] = True  # 确保 SVC 输出概率
            clf_inner = SVC(**best_params)
            clf_inner.fit(X_train_inner, y_train_inner)
            y_pred_prob_inner = clf_inner.predict_proba(X_test_inner)[:, 1]
            inner_metrics_list.append(np.array([y_test_inner, y_pred_prob_inner]))

        # 对外层交叉验证的数据进行评估
        best_params['probability'] = True
        clf_outer = SVC(**best_params)
        clf_outer.fit(X_train_outer, y_train_outer)
        y_pred_prob_outer = clf_outer.predict_proba(X_test_outer)[:, 1]
        out_metrics_list.append(np.array([y_test_outer, y_pred_prob_outer]))

    # 在整个训练集上训练 SVM 模型
    best_params['probability'] = True
    clf_final = SVC(**best_params)
    clf_final.fit(X_train_arr, y_train)

    # 使用 CalibratedClassifierCV 进行校准
    # 注意：这里由于数据量较少，采用 'prefit' 模式，使用同一训练数据进行校准
    calibrated_clf = CalibratedClassifierCV(estimator=clf_final, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_train_arr, y_train)

    # 保存校准后的模型
    best_model_item = {'clf': calibrated_clf}
    with open(f'./model_history_holdout/{model_name}_{_}.pkl', 'wb') as f:
        pickle.dump(best_model_item, f)

    # 用校准后的模型对保留集进行预测
    y_pred_prob_holdout = calibrated_clf.predict_proba(X_holdout_arr)[:, 1]
    hold_metrics_list.append(np.array([y_holdout, y_pred_prob_holdout]))
    hold_params_list.append(best_params)

    temp_acc = balanced_accuracy_score(y_holdout, (y_pred_prob_holdout > 0.5).astype(int))

    # 更新全局最佳模型（根据保留集准确率）
    if temp_acc > global_best_item.get("eval_acc", 0):
        global_best_item["params"] = best_params
        global_best_item["clf"] = calibrated_clf
        global_best_item["eval"] = hold_metrics_list[-1]
        global_best_item["index"] = _
        global_best_item["eval_acc"] = temp_acc

    # 保存当前 Optuna 的试验记录
    df = study.trials_dataframe()
    df.to_csv(f'./optuna/optuna_{model_name}_{_}.csv', index=False, encoding='utf_8_sig')

global_best_item_list.append(global_best_item)

# -------------------------------
# 4. 保存各项指标和模型
with open('metrics/inner_metrics_list.pkl', 'wb') as f:
    pickle.dump(inner_metrics_list, f)

with open('metrics/out_metrics_list.pkl', 'wb') as f:
    pickle.dump(out_metrics_list, f)

with open('metrics/hold_metrics_list.pkl', 'wb') as f:
    pickle.dump(hold_metrics_list, f)

with open('metrics/best_params_list.pkl', 'wb') as f:
    pickle.dump(best_params_list, f)

with open('metrics/hold_params_list.pkl', 'wb') as f:
    pickle.dump(hold_params_list, f)

with open('metrics/best_model_item.pkl', 'wb') as f:
    pickle.dump(global_best_item, f)

with open('metrics/best_model_item_list.pkl', 'wb') as f:
    pickle.dump(global_best_item_list, f)

print("SVM模型训练全部完成！")