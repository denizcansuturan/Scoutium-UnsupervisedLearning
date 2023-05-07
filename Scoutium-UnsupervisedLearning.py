import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.simplefilter("ignore")


################################
# Business Problem
################################
"""
Predicting the class (average, highlighted) of football players based on the ratings given to their characteristics 
observed by scouts.

The dataset consists of information from Scoutium on football players evaluated by scouts based on their 
characteristics observed during matches, including the scored features and their ratings during the matches.
"""
################################
# STEP 1: Reading the data
################################

attributes = pd.read_csv("machine_learning/PART3/scoutium_attributes.csv", sep=";") #, index_col=0
potential_labels = pd.read_csv("machine_learning/PART3/scoutium_potential_labels.csv", sep=";")
attributes.head()
potential_labels.head()
attributes["attribute_id"].unique()

################################
# STEP 2: Merging the CSV files
################################

dff = pd.merge(attributes, potential_labels, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
dff.tail()

################################
# STEP 3: Removing the goalkeeper (1) class from the dataset?
################################

dff = dff[dff["position_id"] != 1]

################################
# STEP 4: Removing the below_average class from the potential_label (which constitutes 1% of the entire dataset).
################################

dff["potential_label"].unique()
dff["potential_label"].value_counts()

dff = dff[dff["potential_label"] != "below_average"]
dff.shape

################################
# STEP 5: Creating a table from the dataset that is generated using the "pivot_table" function.
# Manipulating it so that each row represents a player in the pivot table.
################################

dff.head()

# a: For each column, including the "position_id", "potential_label", and all "attribute_ids" for each player in order.
pivot_table = pd.pivot_table(dff, values="attribute_value",
                                  columns="attribute_id",
                                  index=["player_id", "position_id","potential_label"])

pivot_table.iloc[:5, :5]

# b: Using the "reset_index" function to fix any indexing errors and convert the column names of "attribute_id"
# to strings using df.columns.map(str).

pivot_table = pivot_table.reset_index(drop=False)
pivot_table = pivot_table.rename_axis(columns=None)
pivot_table.columns = pivot_table.columns.map(str)

################################
# STEP 6: Using the LabelEncoder function to numerically encode the categories of "potential_label" (average, highlighted).
################################

le = LabelEncoder()
pivot_table["potential_label"] = le.fit_transform(pivot_table["potential_label"])

################################
# STEP 7: Saving the numerical variable columns as a list named "num_cols".
################################

num_cols = pivot_table.columns[4:]

################################
# STEP 8: Applying the StandardScaler to scale the data in all of the variables saved in the "num_cols" list that is saved earlier.
################################

scaler = StandardScaler()
pivot_table[num_cols] = scaler.fit_transform(pivot_table[num_cols])

pivot_table.iloc[:5, :5]

################################
# STEP 9: Developing a machine learning model that predicts the potential labels of football players in our dataset with minimum error.
################################

y = pivot_table["potential_label"]
X = pivot_table.drop(["potential_label", "player_id"], axis=1)

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ("SVC", SVC()),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ('Adaboost', AdaBoostClassifier()),
          ('GBM', GradientBoostingClassifier()),
          ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
          ('CatBoost', CatBoostClassifier(verbose=False)),
          ("LightGBM", LGBMClassifier())]

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=5).mean()
        print(score+" score:"+str(round(cvs, 4)))
    print("\n\n")

"""
LR
roc_auc score:0.8354
f1 score:0.5783
precision score:0.6867
recall score:0.5167
accuracy score:0.8488

KNN
roc_auc score:0.7864
f1 score:0.348
precision score:0.5367
recall score:0.2803
accuracy score:0.8044

SVC
roc_auc score:0.8342
f1 score:0.0
precision score:0.0
recall score:0.0
accuracy score:0.7934

CART
roc_auc score:0.7203
f1 score:0.5581
precision score:0.5273
recall score:0.5318
accuracy score:0.8046

RF
roc_auc score:0.9041
f1 score:0.5622
precision score:0.8981
recall score:0.4424
accuracy score:0.8671

Adaboost
roc_auc score:0.8467
f1 score:0.6224
precision score:0.7109
recall score:0.5848
accuracy score:0.8562

GBM
roc_auc score:0.8784
f1 score:0.5863
precision score:0.7862
recall score:0.5136
accuracy score:0.8562

XGBoost
roc_auc score:0.8541
f1 score:0.6078
precision score:0.733
recall score:0.55
accuracy score:0.8562

CatBoost
roc_auc score:0.8912
f1 score:0.5852
precision score:0.8933
recall score:0.4606
accuracy score:0.8709

LightGBM
roc_auc score:0.883
f1 score:0.6475
precision score:0.7705
recall score:0.5864
accuracy score:0.8709

"""

################################
# STEP 10: Using the feature_importance function to determine the importance levels of variables and plot the ranking of features.
################################


def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")


model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)

X.columns
random_user = X.sample(1, random_state=45)
model.predict(random_user)
pivot_table.iloc[162]