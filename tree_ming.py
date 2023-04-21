import os
import graphviz
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import dtreeviz

import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree, DecisionTreeClassifier, plot_tree, export_graphviz


warnings.filterwarnings("ignore")
pd.set_option('display.width', 5000)
plt.style.use('seaborn-ticks')
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


def get_dt_rules(tree, feature_names, total_bad_rate, total_count):
    tree_ = tree.tree_
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    feature_name = [feature_names[i] if i != -2 else "undefined!" for i in tree_.feature]
    rules=dict()

    global res_df
    res_df = pd.DataFrame()
    
    def recurse(node, depth, parent): # 搜每个节点的规则

        if tree_.feature[node] != -2:  # 非叶子节点,搜索每个节点的规则
            name = feature_name[node]
            thd = np.round(tree_.threshold[node],3)
            s= "{} <= {} ".format( name, thd, node )
            # 左子
            if node == 0:
                rules[node]=s
            else:
                rules[node]=rules[parent]+' & ' +s
            recurse(left[node], depth + 1, node)
            s="{} > {}".format(name, thd)
            # 右子 
            if node == 0:
                rules[node]=s
            else:
                rules[node]=rules[parent]+' & ' +s
            recurse(right[node], depth + 1, node)
        else:
            df = pd.DataFrame()
            df['组合策略'] = rules[parent],
            df['好样本数'] = tree_.value[node][0][0].astype(int)
            df['好样本占比'] = df['好样本数'] / (total_count * (1 - total_bad_rate))
            df['坏样本数'] = tree_.value[node][0][1].astype(int)
            df['坏样本占比'] = df['坏样本数'] / (total_count * total_bad_rate)
            df['命中数'] = df['好样本数'] + df['坏样本数']
            df['命中率'] = df['命中数'] / total_count
            df['坏率'] = df['坏样本数'] / df['命中数']
            df['样本整体坏率'] = total_bad_rate
            df['LIFT值'] = df['坏率'] / df['样本整体坏率']
            
            global res_df
            
            res_df = pd.concat([res_df, df], 0)
            
    recurse(0, 1, 0)
    
    return res_df.sort_values("LIFT值", ascending=True).reset_index(drop=True)


def dtreeviz_plot(tree, X_TE, y, target="target", save=None):
    viz_model = dtreeviz.model(tree,
                               X_train=X_TE, y_train=y,
                               feature_names=X_TE.columns,
                               target_name=target, class_names=["GOOD", f"BAD"])
    viz = viz_model.view(
                scale=1.5, 
                orientation='LR', 
                colors={
                        "classes": [None, None, ["#2639E9", "#F76E6C"], ["#2639E9", "#F76E6C", "#FE7715", "#FFFFFF"]],
                        "arrow": "#2639E9",
                        'text_wedge': "#F76E6C",
                        "pie": "#2639E9",
                        "tile_alpha": 1,
                        "legend_edge": "#FFFFFF",
                    },
                ticks_fontsize=10,
                label_fontsize=10,
            )
    
#     viz = dtreeviz.model(
#         decision_tree,
#         X_TE,
#         y,
#         # title="DecisionTreeClassifier",
#         # title_fontsize=10,
#         ticks_fontsize=10,
#         label_fontsize=10,
#         target_name=target,
#         feature_names=X_TE.columns,
#         class_names=["good", "bad"],
#         orientation='LR',
#         scale=1.5,
#         colors={
#             "classes": [None, None, ["#2639E9", "#F76E6C"], ["#2639E9", "#F76E6C", "#FE7715", "#FFFFFF"]],
#             "arrow": "#2639E9",
#             'text_wedge': "#F76E6C",
#             "pie": "#2639E9",
#             "tile_alpha": 1,
#             "legend_edge": "#FFFFFF",
#         },
#     )
    
    if save:
        viz.save(save)
    
    return viz


if __name__ == '__main__':
    import scorecardpy as sc
    
    target = "creditability"
    data = sc.germancredit()
    data[target] = data[target].map({"good": 0, "bad": 1})
    
    cat_features = list(set(data.select_dtypes(include=[object, pd.CategoricalDtype]).columns) - set([target]))
    cat_features_index = [i for i, f in enumerate(data.columns) if f in cat_features]

    X = data.drop(columns=[target])
    y = data[target]
    
    target_enc = ce.TargetEncoder(cols=cat_features)
    target_enc.fit(X[cat_features], y)

    X_TE = X.join(target_enc.transform(X[cat_features]).add_suffix('_target'))

    target_enc.target_mapping = {}
    for col in cat_features:
        mapping = X_TE[[col, f"{col}_target"]].drop_duplicates()
        target_enc.target_mapping[col] = dict(zip(mapping[col], mapping[f"{col}_target"]))

    X_TE = X_TE.drop(columns=cat_features)
    X_TE = X_TE.rename(columns={f"{c}_target": c for c in cat_features})
    
    removes = []
    dt_rules = pd.DataFrame()
    
    for i in range(128):
        decision_tree = DecisionTreeClassifier(max_depth=2, min_samples_split=8, min_samples_leaf=5, max_features="auto")
        decision_tree = decision_tree.fit(X_TE, y)

        if decision_tree.score(X_TE, y) < 0.8:
            break

        rules = get_dt_rules(decision_tree, X_TE.columns, sum(y) / len(y), len(y))
        viz_model = dtreeviz.model(decision_tree,
                                   X_train=X_TE, y_train=y,
                                   feature_names=X_TE.columns,
                                   target_name=target, class_names=["DPD 0", f"DPD {dpd}+"])

        rules = rules.query("LIFT值 > 4 & 命中率 < 0.1")

        if len(rules) > 0:
            print("/" * 150)
            rules["组合策略"] = rules["组合策略"].replace(feature_map, regex=True)
            display(rules)
            c = viz_model.view(
                scale=1.5, 
                orientation='LR', 
                colors={
                        "classes": [None, None, ["#2639E9", "#F76E6C"], ["#2639E9", "#F76E6C", "#FE7715", "#FFFFFF"]],
                        "arrow": "#2639E9",
                        'text_wedge': "#F76E6C",
                        "pie": "#2639E9",
                        "tile_alpha": 1,
                        "legend_edge": "#FFFFFF",
                    },
                ticks_fontsize=10,
                label_fontsize=10,
            )
            display(c)

            dt_rules = pd.concat([dt_rules, rules]).reset_index(drop=True)
            removes.append(decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(max(decision_tree.feature_importances_))])
            X_TE = X_TE.drop(columns=removes[-1])
            print("-" * 150)

    pd.set_option('display.max_row', None)
    dt_rules.sort_values(["LIFT值", "命中率"], ascending=False)
    
#     decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
#     decision_tree = decision_tree.fit(X_TE, y)
    
#     rules = get_dt_rules(decision_tree, X_TE.columns, sum(y) / len(y), len(y))
    
#     dtreeviz_plot(decision_tree, X_TE, y, save="decision_tree.svg")
#     rules.to_excel("组合策略挖掘.xlsx")
    
#     dot_data = export_graphviz(decision_tree, feature_names=X_TE.columns, class_names=True, filled=True, rounded=False, out_file=None)
#     graph = graphviz.Source(dot_data)
    
#     graph.render("组合策略挖掘")
