# 可用于 `超参数搜索` & `pipeline` 的逻辑回归

## 概述

分别基于 `statsmodels` 和 `scikit-learn` 实现两种可用于 `sklearn pipeline` 的 `LogisticRegression`，并输出相应的报告，效果如下：

> 基于 `statsmodels` 的 `StatsLogisticRegression`


<img src="https://itlubber.art/upload/2022/10/iShot_2022-10-28_13.21.00.png"></img>
<img src="https://itlubber.art/upload/2022/10/iShot_2022-10-28_13.14.39.png"></img>


> 基于 `sklearn` 的 `ITLubberLogisticRegression`

<img src="https://itlubber.art/upload/2022/11/image-1669653191871.png"></img>

<img src="outputs/logistic_train.png"></img>

<img src="outputs/train_scorehist.png"></img>


## 使用方法

```python
target = "creditability"
data = sc.germancredit()
data[target] = data[target].map({"good": 0, "bad": 1})

train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[target])
oot = data.copy()
feature_pipeline = Pipeline([
    ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
    ("combiner", Combiner(target=target, min_samples=0.2)),
    ("transform", WOETransformer(target=target)),
    ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
    ("stepwise", StepwiseSelection(target=target)),
])

feature_pipeline.fit(train)

woe_train = feature_pipeline.transform(train)
woe_test = feature_pipeline.transform(test)
woe_oot = feature_pipeline.transform(oot)

# logistic = StatsLogisticRegression(target=target)
logistic = ITLubberLogisticRegression(target=target)

logistic.fit(woe_train)

y_pred_train = logistic.predict_proba(woe_train.drop(columns=target))[:, 1]
y_pred_test = logistic.predict_proba(woe_test.drop(columns=target))[:, 1]
y_pred_oot = logistic.predict_proba(woe_oot.drop(columns=target))[:, 1]

# params_grid = {
#     # "logistic__C": [i / 1. for i in range(1, 10, 2)],
#     # "logistic__penalty": ["l2"],
#     # "logistic__class_weight": [None, "balanced"], # + [{1: i / 10.0, 0: 1 - i / 10.0} for i in range(1, 10)],
#     # "logistic__max_iter": [100],
#     # "logistic__solver": ["sag"] # ["liblinear", "sag", "lbfgs", "newton-cg"],
#     "logistic__intercept": [True, False],
# }

# clf = GridSearchCV(feature_pipeline, params_grid, cv=5, scoring='roc_auc', verbose=-1, n_jobs=2, return_train_score=True)
# clf.fit(train, train[target])

# y_pred_train = clf.best_estimator_.predict(train)
# y_pred_test = clf.best_estimator_.predict(test)

# print(clf.best_params_)

# model summary
# logistic.summary_save()
# logistic.plot_weights(save="logistic_train.png")
summary = logistic.summary().reset_index().rename(columns={"index": "Features"})

train_report = logistic.report(woe_train)
test_report = logistic.report(woe_test)
oot_report = logistic.report(woe_oot)

print("train: ", toad.metrics.KS(y_pred_train, train[target]), toad.metrics.AUC(y_pred_train, train[target]))
print("test: ", toad.metrics.KS(y_pred_test, test[target]), toad.metrics.AUC(y_pred_test, test[target]))
print("oot: ", toad.metrics.KS(y_pred_oot, oot[target]), toad.metrics.AUC(y_pred_oot, oot[target]))

card = ScoreCard(target=target, pipeline=feature_pipeline, pretrain_lr=logistic)
card.fit(woe_train)

train["score"] = card.predict(train)
test["score"] = card.predict(test)
oot["score"] = card.predict(oot)

# print(card.feature_bin_stats(train, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step"))
# print(card.feature_bin_stats(train, "score", target=target, verbose=0, method="cart"))

train_score_rank = card.feature_bin_stats(train, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step")
test_score_rank = card.feature_bin_stats(test, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step")
oot_score_rank = card.feature_bin_stats(oot, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step")

writer = pd.ExcelWriter("评分卡结果验证表.xlsx", engine="openpyxl")

summary.to_excel(writer, sheet_name="逻辑回归拟合结果", startrow=1, index=False)
train_report.to_excel(writer, sheet_name="逻辑回归拟合结果", startrow=len(summary) + 5, index=False)
test_report.to_excel(writer, sheet_name="逻辑回归拟合结果", startrow=len(summary) + len(train_report) + 9, index=False)
oot_report.to_excel(writer, sheet_name="逻辑回归拟合结果", startrow=len(summary) + len(train_report) + len(test_report) + 13, index=False)

worksheet = writer.sheets['逻辑回归拟合结果']
worksheet.cell(row=1, column=1).value = "入模变量系数及相关统计指标"
worksheet.cell(row=len(summary) + 5, column=1).value = "训练数据集模型预测报告"
worksheet.cell(row=len(summary) + len(train_report) + 9, column=1).value = "测试数据集模型预测报告"
worksheet.cell(row=len(summary) + len(train_report) + len(test_report) + 13, column=1).value = "跨时间验证集模型预测报告"

train_score_rank.to_excel(writer, sheet_name="评分卡排序性", startrow=1, index=False)
test_score_rank.to_excel(writer, sheet_name="评分卡排序性", startrow=len(train_score_rank) + 5, index=False)
oot_score_rank.to_excel(writer, sheet_name="评分卡排序性", startrow=len(train_score_rank) + len(test_score_rank) + 9, index=False)

worksheet = writer.sheets['评分卡排序性']

worksheet.cell(row=1, column=1).value = "训练数据集评分排序性"
worksheet.cell(row=len(train_score_rank) + 5, column=1).value = "测试数据集评分排序性"
worksheet.cell(row=len(train_score_rank) + len(test_score_rank) + 9, column=1).value = "跨时间验证集评分排序性"

writer.close()

from utils import render_excel

render_excel("评分卡结果验证表.xlsx", border=False)
```


## 参考

> https://github.com/ing-bank/skorecard/blob/main/skorecard/linear_model/linear_model.py
> 
> https://github.com/itlubber/openpyxl-excel-style-template/blob/main/pipeline_model.py
> 