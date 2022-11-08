# 可用于 `超参数搜索` & `pipeline` 的逻辑回归

## 概述

分别基于 `statsmodels` 和 `scikit-learn` 实现两种可用于 `sklearn pipeline` 的 `LogisticRegression`，并输出相应的报告，效果如下：

> 基于 `statsmodels` 的 `StatsLogisticRegression`


<img src="https://itlubber.art/upload/2022/10/iShot_2022-10-28_13.21.00.png"></img>
<img src="https://itlubber.art/upload/2022/10/iShot_2022-10-28_13.14.39.png"></img>


> 基于 `sklearn` 的 `ITLubberLogisticRegression`


<img src="https://itlubber.art/upload/2022/10/iShot_2022-10-28_13.16.32.png"></img>


## 使用方法

```python
target = "creditability"
data = sc.germancredit()
data[target] = data[target].map({"good": 0, "bad": 1})

train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[target])

# 定义 pipeline
feature_pipeline = Pipeline([
    ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
    ("combiner", Combiner(target=target, min_samples=0.2)),
    ("transform", WOETransformer(target=target)),
    ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
    ("stepwise", StepwiseSelection(target=target)),
    ("logistic", StatsLogisticRegression(target=target)),
    # ("logistic", ITLubberLogisticRegression(target=target)),
])

# 训练 & 预测
feature_pipeline.fit(train)
y_pred_train = feature_pipeline.predict(train.drop(columns=target))
y_pred_test = feature_pipeline.predict(test.drop(columns=target))

# StatsLogisticRegression 参数搜索
params_grid = {
    "logistic__intercept": [True, False],
}

# ITLubberLogisticRegression 参数搜索
# params_grid = {
#     "logistic__C": [i / 1. for i in range(1, 10, 2)],
#     "logistic__penalty": ["l2"],
#     "logistic__class_weight": [None, "balanced"], # + [{1: i / 10.0, 0: 1 - i / 10.0} for i in range(1, 10)],
#     "logistic__max_iter": [100],
#     "logistic__solver": ["sag"] # ["liblinear", "sag", "lbfgs", "newton-cg"],
# }

clf = GridSearchCV(feature_pipeline, params_grid, cv=5, scoring='roc_auc', verbose=-1, n_jobs=2, return_train_score=True)
clf.fit(train, train[target])

y_pred_train = clf.best_estimator_.predict(train)
y_pred_test = clf.best_estimator_.predict(test)

print(clf.best_params_)

# model summary
feature_pipeline.named_steps['logistic'].summary_save()
# feature_pipeline.named_steps['logistic'].summary()

print("train: ", toad.metrics.KS(y_pred_train, train[target]), toad.metrics.AUC(y_pred_train, train[target]))
print("test: ", toad.metrics.KS(y_pred_test, test[target]), toad.metrics.AUC(y_pred_test, test[target]))
```


## 参考

> https://github.com/ing-bank/skorecard/blob/main/skorecard/linear_model/linear_model.py
> 
> https://github.com/itlubber/openpyxl-excel-style-template/blob/main/pipeline_model.py
> 