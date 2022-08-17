## Importing and Settings
import pandas as pd
import numpy as np
import math
import scipy.stats as st
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from scipy.stats._stats_py import ttest_ind
import matplotlib as mt
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from datetime import date
# Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
# Recomendation Systems
from mlxtend.frequent_patterns import apriori, association_rules
# Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
# Feature Engineering
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

## Business Problem
# Goal is building a ML algorithm that can predict individuals diabities conditions when necceseary data provided.
# Its expected to conduct neceseary data analysis and feature engineering steps.

## Data
df = pd.read_csv('/Users/buraksayilar/Desktop/feature_engineering/feature_engineering/datasets/diabetes.csv')

def check_df(dataframe, head=5):
    print(f'{" Shape ":-^100}')
    print(dataframe.shape)
    print(f'{" Info":-^100}')
    print(dataframe.info(head))
    print(f'{" Head ":-^100}')
    print(dataframe.head(head))
    print(f'{" Tail ":-^100}')
    print(dataframe.tail(head))
    print(f'{" NA ":-^100}')
    print(dataframe.isnull().sum())
    print(f'{" Quantiles ":-^100}')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
# Is seems there is some outlier values in Insulin and maybe in Age and SkinThiknes.
df.columns = [col.upper() for col in df.columns]

## Explanetory Data Analysis
def advance_histogram(df):
    plt.figure(figsize=(10,8))
    i = 1
    for col_name in df.columns:
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=col_name)
        i += 1
    plt.show()
advance_histogram(df)
def target_variable_distribution(data):
    trace = plt.pie(data['OUTCOME'].value_counts(),labels=['healthy', 'diabetic'], autopct='%.0f%%')

    layout = dict(title='Distribution of Target Variable (OUTCOME)')

    fig = dict(data=[trace], layout=layout)
    plt.show()
target_variable_distribution(df)

# Extracting Numerical and Categorical Variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, num_but_car = grab_col_names(df)

# Analysing Numerical and Categorical Variables
# Numerical
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    print(num_summary(df, col))
# Categorical
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    print(cat_summary(df, col))
# Target Summary with Numerical Columns
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
for col in num_cols:
    target_summary_with_num(df, 'OUTCOME',col)
# Target Summary with Categorical Columns
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# Outlier Analysis
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + interquantile_range * 1.5
    low_limit = quantile1 - interquantile_range * 1.5
    return low_limit, up_limit
def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name,q3=0.99)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
       return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(df, col))
def grab_outliers(dataframe, col_name, index=False):
        low, up = outlier_thresholds(dataframe, col_name)
        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
            print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head())
        else:
            print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
            if index:
                outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
                return outlier_index
grab_outliers(df, 'SKINTHICKNESS')
df['INSULIN'].sort_values(ascending=True)
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(df[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(df[col_name] > up_limit), col_name] = up_limit

replace_with_thresholds(df, 'SKINTHICKNESS')

# Missing Value Analysis
# 3 methods are using for missing values;
# Erase, giving values like mod-median and prediction methods
# using with ML.

df.isnull().any()
df.isnull().sum()
df.notnull().sum()
df[df.isnull().any(axis=1)]
# Nulls in percantage
df.isnull().sum() / df.shape[0] *100
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
missing_values_table(df)
# missing table
#!Relation Between Missing Values and Dependent Features
# Filling numerical values directly with median.
# Filling categorical values directly with mode.
# Filling numerical values with respect to categorical value relation.
# Filling with predicted values. (ML)
missing_values_table(df, True)
na_cols = missing_values_table(df, True)
# We took missing values indexes.

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
missing_vs_target(df,'OUTCOME', na_cols)

# Correlation Analysis
sns.heatmap(df.corr(), cmap='RdBu')


## FEATURE ENGINEERING
# Extracting NaN values
for col in num_cols:
    print((df[col] == 0).value_counts())
    nan_columns = ['GLUCOSE', 'BLOODPRESSURE', 'INSULIN', 'BMI', 'SKINTHICKNESS', 'DIABETESPEDIGREEFUNCTION']
#for col in num_cols:
  #  df.replace(0, np.nan,inplace=True)
missing_values_table(df)

# Now we have our missing values. Now we can deal with them.
df.plot.scatter('SKINTHICKNESS', 'BMI')
df["SKINTHICKNESS"].fillna(df.groupby("SKINTHICKNESS")["BMI"].mean()).isnull().sum()

from sklearn.impute import KNNImputer
# KNN method says 'bana arkadaşını söyle sana kim olduğunu söyleyeyim'.
# KNN method takes data from NA values neighbors. And fill
# NA's with these neighbours mean.
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
df[['GLUCOSE', 'BLOODPRESSURE','INSULIN','BMI','SKINTHICKNESS','DIABETESPEDIGREEFUNCTION']] = \
    dff[['GLUCOSE', 'BLOODPRESSURE','INSULIN','BMI','SKINTHICKNESS','DIABETESPEDIGREEFUNCTION']]
missing_values_table(df)


# Feature Extraction (Generating New Features)
# Feature Interactions (Finding New Relations)
df["NEW_INSULIN_FLAG"] = df['INSULIN'].null
df.groupby("OUTCOME")["NEW_INSULIN_FLAG"].mean()
df['NEW_INSULIN_FLAG'] = df['INSULIN'].notnull().astype('int')
target_summary_with_num(df, 'OUTCOME', 'NEW_INSULIN_FLAG')



# Standardization
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

# Creating Model
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


