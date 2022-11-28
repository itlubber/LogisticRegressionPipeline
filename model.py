# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/23 13:12
@Author  : itlubber
@Site    : itlubber.art
"""

import os
import toad
import warnings
import numpy as np
import pandas as pd
import scorecardpy as sc
from optbinning import OptimalBinning
import matplotlib.pyplot as plt
from matplotlib import font_manager
import plotly.graph_objects as go
from openpyxl import load_workbook
from openpyxl.styles import Alignment, PatternFill

import scipy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from processing import FeatureSelection, Combiner, WOETransformer, StepwiseSelection


warnings.filterwarnings("ignore")
pd.set_option('display.width', 5000)
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


class StatsLogisticRegression(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", intercept=True, ):
        self.intercept = intercept
        self.target = target
        self.classifier = None
        self.corr = None
        self.vif = None
        self.coef_normalization = None
        self.feature_names_ = None
        self.feature_importances_ = None
    
    def fit(self, x, y=None, vif=True, corr=True, normalization=True):
        self.feature_names_ = list(x.drop(columns=[self.target]).columns)
        self.feature_importances_ = self.feature_importances(x)
        
        if vif:
            self.vif = self.VIF(x)
            
        if normalization:
            _x = x.drop(columns=[self.target]).apply(lambda x: (x - np.mean(x)) / np.std(x))
            _y = x[self.target]
            lr_normalization = sm.Logit(_y, sm.add_constant(_x) if self.intercept else _x).fit()
            self.coef_normalization = pd.DataFrame(lr_normalization.params, columns=["coef_normalization"])
            
        if corr:
            self.corr = x.drop(columns=[self.target]).corr()
            
        if self.intercept:
            x = sm.add_constant(x)
        
        self.classes_ = x[self.target].unique()
        self.classifier = sm.Logit(x[self.target], x.drop(columns=[self.target])).fit()
        
        return self
    
    def transform(self, x):
        if self.intercept:
            x = sm.add_constant(x)
        
        return self.classifier.predict(x)
    
    def predict(self, x):
        return self.transform(x)
    
    def summary(self):
        describe = self.classifier.summary2()
        return describe
    
    def feature_importances(self, x):
        params = {
            "n_estimators": 256,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 1e-3,
            "loss": "deviance",
            "subsample": 0.9,
        }
        feature_importances_ = GradientBoostingClassifier(**params).fit(x.drop(columns=[self.target]), x[self.target]).feature_importances_
        return pd.DataFrame(feature_importances_, index=self.feature_names_, columns=["feature_importances"])
        
    def VIF(self, x):
        if self.intercept:
            x = sm.add_constant(x)
        
        x = x.drop(columns=[self.target])
        columns = x.columns
        vif = pd.DataFrame({"VIF": [variance_inflation_factor(np.matrix(x), i) for i in range(len(columns))]}, index=columns)
        
        return vif
    
    def WALD(self):
        return self.classifier.wald_test_terms().table[["statistic", "pvalue"]].rename(columns={"pvalue": "wald_test_pvalue", "statistic": "wald_test_statistic"})
    
    def report(self):
        return self.classifier.summary2().tables[1].join([self.coef_normalization, self.WALD(), self.vif, self.feature_importances_]), self.classifier.summary2().tables[0], self.corr
    
    def summary_save(self, excel_name="逻辑回归模型拟合效果.xlsx", sheet_name="逻辑回归拟合效果"):
        writer = pd.ExcelWriter(excel_name, engine='openpyxl')
        
        coef_report, summary_report, corr_report = self.report()
        summary_report.columns = ["逻辑回归模型拟合效果"] * summary_report.shape[1]
        summary_report.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startcol=0, startrow=2)
        coef_report.reset_index().rename(columns={"index": "variable"}).to_excel(writer, sheet_name=sheet_name, index=False, header=True, startcol=0, startrow=summary_report.shape[0] + 4)
        corr_report.to_excel(writer, sheet_name=sheet_name, index=True, header=True, startcol=0, startrow=summary_report.shape[0] + coef_report.shape[0] + 7)
        
        writer.save()
        writer.close()
        
        if os.path.exists(excel_name):
            workbook = load_workbook(excel_name)
            worksheet = workbook.get_sheet_by_name(sheet_name)
            worksheet["A1"].value = "逻辑回归模型报告"
            worksheet["A1"].alignment = Alignment(horizontal='center', vertical='center')
            worksheet.merge_cells(f"A1:L1")
            
            workbook.save(excel_name)
            workbook.close()
        
        try:
            from processing import render_excel # From: https://github.com/itlubber/openpyxl-excel-style-template/blob/main/feature_bins.py
            render_excel(excel_name, sheet_name=sheet_name, max_column_width=25, merge_rows=np.cumsum([1, len(summary_report), 2, len(coef_report) + 1, 2, len(corr_report) + 1]).tolist())
        except:
            pass


class ITLubberLogisticRegression(LogisticRegression):
    """
    Extended Logistic Regression.
    Extends [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
    This class provides the following extra statistics, calculated on `.fit()` and accessible via `.summary()`:
    - `cov_matrix_`: covariance matrix for the estimated parameters.
    - `std_err_intercept_`: estimated uncertainty for the intercept
    - `std_err_coef_`: estimated uncertainty for the coefficients
    - `z_intercept_`: estimated z-statistic for the intercept
    - `z_coef_`: estimated z-statistic for the coefficients
    - `p_value_intercept_`: estimated p-value for the intercept
    - `p_value_coef_`: estimated p-value for the coefficients
    
    Example:
    ```python
    feature_pipeline = Pipeline([
        ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("combiner", Combiner(target=target, min_samples=0.2)),
        ("transform", WOETransformer(target=target)),
        ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("stepwise", StepwiseSelection(target=target)),
        # ("logistic", LogisticClassifier(target=target)),
        ("logistic", ITLubberLogisticRegression(target=target)),
    ])
    
    feature_pipeline.fit(train)
    summary = feature_pipeline.named_steps['logistic'].summary()
    ```
    
    An example output of `.summary()`:
    
    |                   |     Coef. |   Std.Err |        z |       P>|z| |    [ 0.025 |   0.975 ] |     VIF |
    |:------------------|----------:|----------:|---------:|------------:|-----------:|----------:|--------:|
    | const             | -0.844037 | 0.0965117 | -8.74544 | 2.22148e-18 | -1.0332    | -0.654874 | 1.05318 |
    | duration.in.month |  0.847445 | 0.248873  |  3.40513 | 0.000661323 |  0.359654  |  1.33524  | 1.14522 |
    """

    def __init__(self, target="target", penalty="l2", calculate_stats=True, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,):
        """
        Extends [sklearn.linear_model.LogisticRegression.fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
        Args:
            calculate_stats (bool): If true, calculate statistics like standard error during fit, accessible with .summary()
        """
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio,)
        self.target = target
        self.calculate_stats = calculate_stats

    def fit(self, x, sample_weight=None, **kwargs):
        y = x[self.target]
        x = x.drop(columns=[self.target])
        
        if not self.calculate_stats:
            return super().fit(x, y, sample_weight=sample_weight, **kwargs)

        x = self.convert_sparse_matrix(x)
        
        if isinstance(x, pd.DataFrame):
            self.names_ = ["const"] + [f for f in x.columns]
        else:
            self.names_ = ["const"] + [f"x{i}" for i in range(x.shape[1])]

        lr = super().fit(x, y, sample_weight=sample_weight, **kwargs)

        predProbs = self.predict_proba(x)

        # Design matrix -- add column of 1's at the beginning of your x matrix
        if lr.fit_intercept:
            x_design = np.hstack([np.ones((x.shape[0], 1)), x])
        else:
            x_design = x

        self.vif = [variance_inflation_factor(np.matrix(x_design), i) for i in range(x_design.shape[-1])]
        p = np.product(predProbs, axis=1)
        self.cov_matrix_ = np.linalg.inv((x_design * p[..., np.newaxis]).T @ x_design)
        std_err = np.sqrt(np.diag(self.cov_matrix_)).reshape(1, -1)

        # In case fit_intercept is set to True, then in the std_error array
        # Index 0 corresponds to the intercept, from index 1 onwards it relates to the coefficients
        # If fit intercept is False, then all the values are related to the coefficients
        if lr.fit_intercept:

            self.std_err_intercept_ = std_err[:, 0]
            self.std_err_coef_ = std_err[:, 1:][0]

            self.z_intercept_ = self.intercept_ / self.std_err_intercept_

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = scipy.stats.norm.sf(abs(self.z_intercept_)) * 2

        else:
            self.std_err_intercept_ = np.array([np.nan])
            self.std_err_coef_ = std_err[0]

            self.z_intercept_ = np.array([np.nan])

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = np.array([np.nan])

        self.z_coef_ = self.coef_ / self.std_err_coef_
        self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_)) * 2

        return self

    @staticmethod
    def report(woe_train):
        return pd.DataFrame(classification_report(train[target], logistic.predict(woe_train.drop(columns=target)), output_dict=True)).T.reset_index().rename(columns={"index": "desc"})

    def summary(self):
        """
        Puts the summary statistics of the fit() function into a pandas DataFrame.
        Returns:
            data (pandas DataFrame): The statistics dataframe, indexed by the column name
        """
        check_is_fitted(self)

        if not hasattr(self, "std_err_coef_"):
            msg = "Summary statistics were not calculated on .fit(). Options to fix:\n"
            msg += "\t- Re-fit using .fit(X, y, calculate_stats=True)\n"
            msg += "\t- Re-inititialize using LogisticRegression(calculate_stats=True)"
            raise AssertionError(msg)

        data = {
            "Coef.": (self.intercept_.tolist() + self.coef_.tolist()[0]),
            "Std.Err": (self.std_err_intercept_.tolist() + self.std_err_coef_.tolist()),
            "z": (self.z_intercept_.tolist() + self.z_coef_.tolist()[0]),
            "P>|z|": (self.p_val_intercept_.tolist() + self.p_val_coef_.tolist()[0]),
        }
        
        stats = pd.DataFrame(data, index=self.names_)
        stats["[ 0.025"] = stats["Coef."] - 1.96 * stats["Std.Err"]
        stats["0.975 ]"] = stats["Coef."] + 1.96 * stats["Std.Err"]
        
        stats["VIF"] = self.vif
        
        return stats
    
    @staticmethod
    def convert_sparse_matrix(x):
        """
        Converts a sparse matrix to a numpy array.
        This can prevent problems arising from, e.g. OneHotEncoder.
        Args:
            x: numpy array, sparse matrix
        Returns:
            numpy array of x
        """
        if scipy.sparse.issparse(x):
            return x.toarray()
        else:
            return x

    def plot_weights(self):
        """
        Generates a weight plot(plotly chart) from `stats`
        Example:
        ```
        pipeline = Pipeline([
            ('clf', LogisticRegression(calculate_stats=True))
        ])
        pipeline.fit(X, y)
        stats = pipeline.named_steps['clf'].plot_weights()
        ```
        Args:
            stats: The statistics to display
            format: The format of the image, such as 'png'. The default None returns a plotly image.
            scale: If format is specified, the scale of the image
            width: If format is specified, the width of the image
            height: If format is specified, the image of the image
        """
        stats = self.summary()
        
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=stats['Coef.'],
                y=stats['Coef.'].index,
                line=dict(color='#2639E9', width=2),
                mode='markers',

                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=stats['0.975 ]'] - stats['Coef.'],
                    arrayminus=stats['Coef.'] - stats['[ 0.025'],
                    color='#2639E9')
            )
        )

        fig.add_shape(type="line",
                    x0=0, y0=0, x1=0, y1=len(stats),
                    line=dict(color="#a29bfe", width=3, dash='dash')
                    )

        fig.update_layout(
            title='Regression Meta Analysis - Weight Plot',
            xaxis_title='Weight Estimates',
            yaxis_title='Variable',
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        
        fig.update_layout(template="simple_white")

        return fig
    
    
class ScoreCard(toad.ScoreCard, TransformerMixin):
    
    def __init__(self, target="target", pdo=60, rate=2, base_odds=35, base_score=750, combiner={}, transer=None, pretrain_lr=None, pipeline=None, **kwargs):
        if pipeline:
            combiner = self.class_steps(pipeline, Combiner)[0]
            transer = self.class_steps(pipeline, WOETransformer)[0]
            
            if self.class_steps(pipeline, (ITLubberLogisticRegression, LogisticRegression)):
                pretrain_lr = self.class_steps(pipeline, (ITLubberLogisticRegression, LogisticRegression))[0]
            
        super().__init__(
                            combiner=combiner.combiner if isinstance(combiner, Combiner) else combiner, transer=transer.transformer if isinstance(transer, WOETransformer) else transer, 
                            pdo=pdo, rate=rate, base_odds=base_odds, base_score=base_score, **kwargs
                        )
        
        self.target = target
        self.pipeline = pipeline
        self.pretrain_lr = pretrain_lr
        
    def fit(self, x):
        y = x[self.target]
        x = x.drop(columns=[self.target])
        
        self._feature_names = x.columns.tolist()

        for f in self.features_:
            if f not in self.transer:
                raise Exception('column \'{f}\' is not in transer'.format(f = f))

        if self.pretrain_lr:
            self.model = self.pretrain_lr
        else:
            self.model.fit(x, y)
        
        self.rules = self._generate_rules()

        sub_score = self.woe_to_score(x)
        self.base_effect = pd.Series(np.median(sub_score, axis=0), index = self.features_)

        return self
    
    def transform(self, x):
        return self.predict(x)
    
    @staticmethod
    def KS_bucket(y_pred, y_true, bucket=10, method="quantile"):
        return toad.metrics.KS_bucket(y_pred, y_true, bucket=bucket, method=method)
    
    @staticmethod
    def KS(y_pred, y_true):
        return toad.metrics.KS(y_pred, y_true)
    
    @staticmethod
    def AUC(y_pred, y_true):
        return toad.metrics.AUC(y_pred, y_true)
    
    @staticmethod
    def perf_eva(y_pred, y_true, title="", plot_type=["ks", "roc"]):
        return sc.perf_eva(y_true, y_pred, title=title, plot_type=plot_type)
    
    @staticmethod
    def ks_plot(y_pred, target, title="", fontsize=14, figsize=(14, 6), save=None, color = ["#2639E9", "#F76E6C", "#FE7715"]):
        fpr, tpr, thresholds = roc_curve(target, y_pred)
        auc_value = auc(fpr, tpr)
        
        fig, ax = plt.subplots(1, 2, figsize = figsize)
        
        # KS曲线
        ax[0].plot(thresholds[1 : ], (tpr - fpr)[1 : ], label = 'Kolmogorov Smirnov', color=color[0])
        ax[0].plot(thresholds[1 : ], tpr[1 : ], label = 'True Positive Rate', color=color[1])
        ax[0].plot(thresholds[1 : ], fpr[1 : ], label = 'False Positive Rate', color=color[2])
        ax[0].fill_between(thresholds[1 : ], fpr[1 : ], tpr[1 : ], color=color[0], alpha=0.25)
        ax[0].tick_params(axis='x', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        ax[0].tick_params(axis='y', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        
        ax[0].spines['top'].set_color(color[0])
        ax[0].spines['bottom'].set_color(color[0])
        ax[0].spines['right'].set_color(color[0])
        ax[0].spines['left'].set_color(color[0])

        ks_value = max(tpr - fpr)
        x = np.argwhere(abs(fpr - tpr) == ks_value)[0, 0]
        thred_value = thresholds[x]
        ax[0].axvline(thred_value, color = color[1], linestyle = ':', ymax = ks_value)
        ax[0].scatter(thred_value, ks_value, c=color[1])
        
        ax[0].set_title(f'KS: {ks_value:.4f}    Best KS Cut Off: {thred_value:.4f}', fontsize=fontsize)
        
        ax[0].set_xlabel("Predict Proba", fontsize=fontsize)
        # ax[0].set_ylabel('Rate')
        
        ax[0].set_xlim((0, max(thresholds[1 : ])))
        ax[0].set_ylim((0, 1))
        
        ax[0].legend(frameon=False, fontsize=fontsize)
        
        # ROC 曲线
        ax[1].plot(fpr, tpr, color=color[0], label="ROC Curve")
        ax[1].stackplot(fpr, tpr, color=color[0], alpha=0.25)
        ax[1].plot([0, 1], [0, 1], color=color[1], lw=2, linestyle=':')
        ax[1].tick_params(axis='x', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        ax[1].tick_params(axis='y', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        
        ax[1].spines['top'].set_color(color[0])
        ax[1].spines['bottom'].set_color(color[0])
        ax[1].spines['right'].set_color(color[0])
        ax[1].spines['left'].set_color(color[0])
        
        ax[1].set_title(f'AUC: {auc_value:.4f}', fontsize=fontsize)

        ax[1].set_xlabel("False Positive Rate", fontsize=fontsize)
        ax[1].set_ylabel('True Positive Rate', fontsize=fontsize)
        
        ax[1].set_xlim((0, 1))
        ax[1].set_ylim((0, 1))
        
        if title: title += " "
        plt.suptitle(f"{title}K-S & ROC CURVE", fontsize=fontsize)
        
        if save:
            plt.savefig(save, dpi=120, format="png", )

        return fig
    
    @staticmethod
    def PSI(y_pred_train, y_pred_oot):
        return toad.metrics.PSI(y_pred_train, y_pred_oot)
    
    @staticmethod
    def perf_psi(y_pred_train, y_pred_oot, y_true_train, y_true_oot, keys=["train", "test"], x_limits=None, x_tick_break=50, show_plot=True, return_distr_dat=False):
        return sc.perf_psi(
            score = {keys[0]: y_pred_train, keys[1]: y_pred_oot},
            label = {keys[0]: y_true_train, keys[1]: y_true_oot},
            x_limits = x_limits,
            x_tick_break = x_tick_break,
            show_plot = show_plot,
            return_distr_dat = return_distr_dat,
        )
    
    @staticmethod
    def score_hist(score, y_true, figsize=(15, 10), bins=20, alpha=0.6):
        mask = y_true == 0
        fig = plt.figure(figsize=figsize)
        plt.hist(score[mask], label="好样本", color="#2639E9", alpha=alpha, bins=bins)
        plt.hist(score[~mask], label="坏样本", color="#F76E6C", alpha=alpha, bins=bins)
        plt.xlabel("score")
        plt.legend()
        # plt.show()
        
        return fig
    
    @staticmethod
    def class_steps(pipeline, query):
        return [v for k, v in pipeline.named_steps.items() if isinstance(v, query)]
    
    @staticmethod
    def format_bins(bins):
        if isinstance(bins, list): bins = np.array(bins)
        EMPTYBINS = len(bins) if not isinstance(bins[0], (set, list, np.ndarray)) else -1
        
        l = []
        if np.issubdtype(bins.dtype, np.number):
            has_empty = len(bins) > 0 and np.isnan(bins[-1])
            if has_empty: bins = bins[:-1]
            sp_l = ["负无穷"] + bins.tolist() + ["正无穷"]
            for i in range(len(sp_l) - 1): l.append('['+str(sp_l[i])+' , '+str(sp_l[i+1])+')')
            if has_empty: l.append('缺失值')
        else:
            for keys in bins:
                keys_update = set()
                for key in keys:
                    if pd.isnull(key) or key == "nan":
                        keys_update.add("缺失值")
                    elif key.strip() == "":
                        keys_update.add("空字符串")
                    else:
                        keys_update.add(key)
                label = ','.join(keys_update)
                l.append(label)

        return {i if b != "缺失值" else EMPTYBINS: b for i, b in enumerate(l)}
    
    def feature_bin_stats(self, data, feature, target="target", rules={}, empty_separate=True, method='step', max_n_bins=10, clip_v=None, desc="评分卡分数", verbose=0):
        if method not in ['dt', 'chi', 'quantile', 'step', 'kmeans', 'cart']:
            raise "method is the one of ['dt', 'chi', 'quantile', 'step', 'kmeans', 'cart']"
        
        combiner = toad.transform.Combiner()
        
        if method == "cart":
            x = data[feature].values
            y = data[target]
            _combiner = OptimalBinning(feature, dtype="numerical", max_n_bins=max_n_bins, monotonic_trend="auto_asc_desc", gamma=0.01).fit(x, y)
            if _combiner.status == "OPTIMAL":
                rules.update({feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.splits] + [np.nan]})
        else:
            combiner.fit(data[[feature, target]], target, empty_separate=empty_separate, method=method, n_bins=max_n_bins, clip_v=clip_v)

        if verbose > 0:
            print(data[feature].describe())

        if rules and isinstance(rules, list): rules = {feature: rules}
        if rules and isinstance(rules, dict): combiner.update(rules)

        feature_bin = combiner.export()[feature]
        feature_bin_dict = self.format_bins(np.array(feature_bin))
        
        df_bin = combiner.transform(data[[feature, target]], labels=False)
        
        table = df_bin[[feature, target]].groupby([feature, target]).agg(len).unstack()
        table.columns.name = None
        table = table.rename(columns = {0 : '好样本数', 1 : '坏样本数'}).fillna(0)
        table["指标名称"] = feature
        table["指标含义"] = desc
        table = table.reset_index().rename(columns={feature: "分箱"})
        table["分箱"] = table["分箱"].map(feature_bin_dict)

        table['样本总数'] = table['好样本数'] + table['坏样本数']
        table['样本占比'] = table['样本总数'] / table['样本总数'].sum()
        table['好样本占比'] = table['好样本数'] / table['好样本数'].sum()
        table['坏样本占比'] = table['坏样本数'] / table['坏样本数'].sum()
        table['坏样本率'] = table['坏样本数'] / table['样本总数']
        
        table = table.fillna(0.)
        
        table['分档WOE值'] = table.apply(lambda x : np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)),axis=1)
        table['分档IV值'] = table.apply(lambda x : (x['好样本占比'] - x['坏样本占比']) * np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)), axis=1)
        table['指标IV值'] = table['分档IV值'].sum()
        
        table["LIFT值"] = table['坏样本率'] / (table["坏样本数"].sum() / table["样本总数"].sum())
        table["累积LIFT值"] = table["LIFT值"].cumsum()
        
        return table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '累积LIFT值']]
    
    
if __name__ == '__main__':
    # https://github.com/itlubber/openpyxl-excel-style-template/blob/main/pipeline_model.py
    
    target = "creditability"
    data = sc.germancredit()
    data[target] = data[target].map({"good": 0, "bad": 1})

    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[target])
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

    # logistic = StatsLogisticRegression(target=target)
    logistic = ITLubberLogisticRegression(target=target)
    
    logistic.fit(woe_train)

    y_pred_train = logistic.predict(woe_train.drop(columns=target))
    y_pred_test = logistic.predict(woe_test.drop(columns=target))
    
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
    print(logistic.summary())
    
    print("train: ", toad.metrics.KS(y_pred_train, train[target]), toad.metrics.AUC(y_pred_train, train[target]))
    print("test: ", toad.metrics.KS(y_pred_test, test[target]), toad.metrics.AUC(y_pred_test, test[target]))

    card = ScoreCard(target=target, pipeline=feature_pipeline, pretrain_lr=logistic)
    card.fit(woe_train)
    
    train["score"] = card.predict(train)
    test["score"] = card.predict(test)
    
    print(card.feature_bin_stats(train, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step"))
    print(card.feature_bin_stats(train, "score", target=target, verbose=0, method="cart"))
    
    train_score_rank = card.feature_bin_stats(train, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step")
    test_score_rank = card.feature_bin_stats(test, "score", target=target, rules=[i for i in range(400, 800, 50)], verbose=0, method="step")
    
    writer = pd.ExcelWriter("评分卡结果验证表.xlsx", engine="openpyxl")
    
    train_score_rank.to_excel(writer, sheet_name="训练集评分卡排序性")
    test_score_rank.to_excel(writer, sheet_name="测试集评分卡排序性")
    
    writer.close()
    