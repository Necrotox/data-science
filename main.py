import dill
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import time


def session_prep(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    df1 = pd.read_csv('data/ga_sessions.csv')
    target = ['sub_car_claim_click', 'sub_car_claim_submit_click',
              'sub_open_dialog_click', 'sub_custom_question_submit_click',
              'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
              'sub_car_request_submit_click']
    df1['client_do_target_action'] = df1['event_action'].apply(lambda x: 1 if x in target else 0)
    df1 = df1.groupby1(['session_id'], as_index=False).agg({'client_do_target_action': 'max'})
    df = pd.merge(df,df1, on='session_id', how='outer', sort=True)
    df = df.drop(
        ['client_id', 'session_id', 'Unnamed: 0', 'hit_page_path', 'visit_date', 'visit_time',
         'hit_number', 'hit_type', 'event_action', 'event_category'], axis=1)
    df = df.rename({'visit_number': 'count_of_action'}, axis=1)
    df.utm_medium = df.utm_medium.replace({'(none)': '(not set)'})
    return df


def Android_iOs_device_os_cange(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    list_for_android = list(df[df['device_os'] == 'Android'].device_brand.unique())
    list_for_android.remove('(not set)')

    list_for_iOS = list(df[df['device_os'] == 'iOS'].device_brand.unique())
    list_for_iOS.remove('(not set)')

    df['device_os'] = np.where((df['device_brand'].isin(list_for_iOS)) & (df['device_os'].isnull()), 'iOS', df['device_os'])
    df['device_os'] = np.where((df['device_brand'].isin(list_for_android)) & (df['device_os'].isnull()), 'Android', df['device_os'])
    df['device_os'] = np.where((df['device_os'].isnull()), '(not set)', df['device_os'])
    return df


def LabelEcoder(df: pd.DataFrame) -> pd.DataFrame:
    list1 = [i for i in df.columns if
             (i.split('_')[0] in 'utm') or (i.split('_')[0] in 'device') or (i.split('_')[0] in 'geo')]
    df[list1] = df[list1].apply(LabelEncoder().fit_transform)
    return df


def prepare_for_ohe(df: pd.DataFrame) -> pd.DataFrame:
    df[[i for i in df.columns if i != 'hit_time']] = df[[i for i in df.columns if i != 'hit_time']].apply(
        lambda x: x.where(x.map(x.value_counts()) > 80))
    return df


def from_float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    for i in df.columns:
        if df[i].dtype == 'float64':
            df[i] = df[i].astype(int)
    return df


def make_standartScaler(df: pd.DataFrame) -> pd.DataFrame:
    data_1 = df[['count_of_action', 'hit_time']]
    std_scaler = StandardScaler()
    std_scaler.fit(data_1)
    std_scaled = std_scaler.transform(data_1)
    list1 = ['count_of_action', 'hit_time']
    list2 = []
    for name in list1:
        std_name = name + '_std'
        list2.append(std_name)
    df[list2] = std_scaled
    return df


def Pipline() -> None:
    df = pd.read_cvs('data/ga_hits.csv')
    df = df.merge(df, session_prep(pd.read_csv('data/ga_sessions.csv'), on='session_id', how='outer' ))

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('concat_with_sessions', FunctionTransformer(session_prep)),
        ('change_os-names', FunctionTransformer(Android_iOs_device_os_cange)),
        ('delete_low_count_categories', FunctionTransformer(prepare_for_ohe)),
        ('column_transformer', column_transformer)
    ])








def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
