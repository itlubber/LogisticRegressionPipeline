import os
import toad
import scipy
import warnings
import numpy as np
import pandas as pd
import scorecardpy as sc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import Image
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from toad.plot import bin_plot, proportion_plot, corr_plot, badrate_plot
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings("ignore")
pd.set_option('display.width', 5000)
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


class FeatureSelection(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", empty=0.95, iv=0.02, corr=0.7, exclude=None, return_drop=True, identical=0.95, remove=None, engine="scorecardpy", target_rm=False):
        self.engine = engine
        self.target = target
        self.empty = empty
        self.identical = identical
        self.iv = iv
        self.corr = corr
        self.exclude = exclude
        self.remove = remove
        self.return_drop = return_drop
        self.target_rm = target_rm
        self.select_columns = None
        self.dropped = None
    
    def fit(self, x, y=None):
        if self.engine == "toad":
            selected = toad.selection.select(x, target=self.target, empty=self.empty, iv=self.iv, corr=self.corr, exclude=self.exclude, return_drop=self.return_drop)
        else:
            selected = sc.var_filter(x, y=self.target, iv_limit=self.iv, missing_limit=self.empty, identical_limit=self.identical, var_rm=self.remove, var_kp=self.exclude, return_rm_reason=self.return_drop)
            
        if self.return_drop and isinstance(selected, dict):
            self.dropped = selected["rm"]
            self.select_columns = list(selected["dt"].columns)
        elif self.return_drop and isinstance(selected, (tuple, list)):
            self.dropped = pd.DataFrame([(feature, reason) for reason, features in selected[1].items() for feature in features], columns=["variable", "rm_reason"])
            self.select_columns = list(selected[0].columns)
        else:
            self.select_columns = list(selected.columns)
        
        if self.target_rm and target in self.select_columns:
            self.select_columns.remove(target)
            
        return self
        
    def transform(self, x, y=None):
        # if self.engine == "toad":
        #     selected = toad.selection.select(x, target=self.target, empty=self.empty, iv=self.iv, corr=self.corr, exclude=self.exclude, return_drop=self.return_drop)
        # else:
        #     selected = sc.var_filter(x, y=self.target, iv_limit=self.iv, missing_limit=self.empty, identical_limit=self.identical, var_rm=self.remove, var_kp=self.exclude, return_rm_reason=self.return_drop)
            
        # if self.return_drop and isinstance(selected, dict):
        #     self.dropped = selected["rm"]
        #     return selected["dt"]
        # elif self.return_drop and isinstance(selected, (tuple, list)):
        #     self.dropped = pd.DataFrame([(feature, reason) for reason, features in selected[1].items() for feature in features], columns=["variable", "rm_reason"])
        #     return selected[0]
        # else:
        #     return selected
        return x[[col for col in self.select_columns if col in x.columns]]
    
    
class Combiner(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", method='chi', empty_separate=False, min_samples=0.05, n_bins=None, rules={}):
        self.combiner = toad.transform.Combiner()
        self.method = method
        self.empty_separate = empty_separate
        self.target = target
        self.min_samples = min_samples
        self.n_bins = n_bins
        self.rules = rules
    
    def fit(self, x, y=None):
        self.combiner.fit(x, y=self.target, method=self.method, min_samples=self.min_samples, n_bins=self.n_bins)
        self.update(self.rules)
        return self
    
    def transform(self, x, y=None, labels=False):
        return self.combiner.transform(x, labels=labels)
    
    def update(self, rules):
        if isinstance(rules, dict):
            self.combiner.update(self.rules)
            
    def bin_plot(self, data, x, rule=None, labels=True, result=False):
        if rule:
            if isinstance(rule, list):
                rule = {x: rule}
            self.combiner.update(rule)
            
        bin_plot(self.combiner.transform(data, labels=labels), x=x, target=self.target)
        
        if result:
            return self.combiner.export()[x]
        
    def proportion_plot(self, x, transform=False, labels=False):
        if transform:
            x = self.combiner.transform(x, labels=labels)
        proportion_plot(x)
        
    def corr_plot(self, data, transform=False, figure_size=(20, 15)):
        if transform:
            data = self.combiner.transform(data, labels=False)
        
        corr_plot(data, figure_size=figure_size)
        
    def badrate_plot(self, data, date_column, feature, labels=True):
        badrate_plot(self.combiner.transform(data[[date_column, feature, self.target]], labels=labels), target=self.target, x=date_column, by=feature)
        
        
class WOETransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", exclude=None):
        self.target = target
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []
        self.transformer = toad.transform.WOETransformer()
        
    def fit(self, x, y=None):
        self.transformer.fit(x.drop(columns=self.exclude + [self.target]), x[self.target])
        return self

    def transform(self, x, y=None):
        return self.transformer.transform(x)
    
    
class StepwiseSelection(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", estimator="ols", direction="both", criterion="aic", max_iter=None, return_drop=True, exclude=None, intercept=True, p_value_enter=0.2, p_remove=0.01, p_enter=0.01, target_rm=False):
        self.target = target
        self.intercept = intercept
        self.p_value_enter = p_value_enter
        self.p_remove = p_remove
        self.p_enter = p_enter
        self.estimator = estimator
        self.direction = direction
        self.criterion = criterion
        self.max_iter = max_iter
        self.return_drop = return_drop
        self.target_rm = target_rm
        self.exclude = exclude
        self.select_columns = None
        self.dropped = None
    
    def fit(self, x, y=None):
        selected = toad.selection.stepwise(x, target=self.target, estimator=self.estimator, direction=self.direction, criterion=self.criterion, exclude=self.exclude, intercept=self.intercept, p_value_enter=self.p_value_enter, 
                                           p_remove=self.p_remove, p_enter=self.p_enter, return_drop=self.return_drop)
        if self.return_drop:
            self.dropped = pd.DataFrame([(col, "stepwise") for col in selected[1]], columns=["variable", "rm_reason"])
            selected = selected[0]
        
        self.select_columns = list(selected.columns)
        
        if self.target_rm and target in self.select_columns:
            self.select_columns.remove(target)
        
        return self
        
    def transform(self, x, y=None):
        return x[[col for col in self.select_columns if col in x.columns]]


if __name__ == "__main__":
    from model import ITLubberLogisticRegression, StatsLogisticRegression, ScoreCard
    
    target = "creditability"
    data = sc.germancredit()
    data[target] = data[target].map({"good": 0, "bad": 1})
    
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[target])
    
    # selection = FeatureSelection(target=target, engine="toad", return_drop=True, corr=0.9, iv=0.01)
    # train = selection.fit_transform(train)
    
    # combiner = Combiner(min_samples=0.2, empty_separate=True, target=target)
    # combiner.fit(train)
    # train = combiner.transform(train)
    
    # transformer = WOETransformer(target=target)
    # train = transformer.fit_transform(train)
    
    # stepwise = StepwiseSelection(target=target)
    # train = stepwise.fit_transform(train)
    
    feature_pipeline = Pipeline([
        ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("combiner", Combiner(target=target, min_samples=0.2)),
        ("transformer", WOETransformer(target=target)),
        ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("stepwise", StepwiseSelection(target=target, target_rm=False)),
        # ("logistic", StatsLogisticRegression(target=target)),
        # ("logistic", ITLubberLogisticRegression(target=target)),
    ])
    
    # feature_pipeline.fit(train)
    # y_pred_train = feature_pipeline.predict(train.drop(columns=target))
    # y_pred_test = feature_pipeline.predict(test.drop(columns=target))

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
    
    # statmodels methods
    # feature_pipeline.named_steps['logistic'].summary_save()
    
    # print("train: ", toad.metrics.KS(y_pred_train, train[target]), toad.metrics.AUC(y_pred_train, train[target]))
    # print("test: ", toad.metrics.KS(y_pred_test, test[target]), toad.metrics.AUC(y_pred_test, test[target]))
    
    woe_train = feature_pipeline.fit_transform(train)
    woe_test = feature_pipeline.transform(test)
    
    # lr = StatsLogisticRegression(target=target)
    # lr.fit(woe_train)
    # lr.summary_save()

    # cols = list(filter(lambda x: x != target, feature_pipeline.named_steps['preprocessing_select'].select_columns))
    
    combiner = feature_pipeline.named_steps['combiner'].combiner
    transformer = feature_pipeline.named_steps['transformer'].transformer
    
    score_card = ScoreCard(target=target, combiner=combiner, transer=transformer, )
    score_card.fit(woe_train)
    
    
    data["score"] = score_card.transform(data)
    
    print(score_card.KS_bucket(data["score"], data[target]))
    pt = score_card.perf_eva(data["score"], data[target], title="train")
    
    sc = score_card.score_hist(data["score"], data[target])
    
    print(score_card.KS(data["score"], data[target]), score_card.AUC(data["score"], data[target]))
    