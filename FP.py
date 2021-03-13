import numpy
import pandas
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_eval(predictions):
    print(classification_report(Y_test, predictions))
    print("confusion matrix :")
    print(confusion_matrix(Y_test, predictions))
    print("R2: \t", r2_score(Y_test, predictions))
    print("RMSE: \t", sqrt(mean_squared_error(Y_test, predictions)))
    print("MAE: \t", mean_absolute_error(Y_test, predictions))


def preprocess(df):
    # reservation_status
    reservation_status_dic = {'Canceled': 0, 'No-Show': 1, 'Check-Out': 2}
    df['reservation_status'] = df['reservation_status'].replace(reservation_status_dic)
    # hotel
    hotel_dic = {'Resort Hotel': 0, 'City Hotel': 1}
    df['hotel'] = df['hotel'].replace(hotel_dic)
    # month
    arrival_date_month_dic = {'January': 0, 'February': 1, 'March': 2, 'April': 3,
                              'May': 4, 'June': 5, 'July': 6, 'August': 7, 'September': 8,
                              'October': 9, 'November': 10, 'December': 11}
    df['arrival_date_month'] = df['arrival_date_month'].replace(arrival_date_month_dic)
    # meal
    meal_dic = {'RO': 0, 'BB': 1, 'HB': 2, 'FB': 3, 'AI': 4, 'SC': 5, 'Undefined': 6}
    df['meal'] = df['meal'].replace(meal_dic)
    # country
    country_dic = {}
    countries_names = df['country'].unique().tolist()
    countries_names.remove(numpy.nan)
    for x in countries_names:
        country_dic[x] = countries_names.index(x)
    df['country'] = df['country'].replace(country_dic)
    # market_segment
    market_segment_dic = {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                          'Complementary': 4, 'Groups': 5, 'Aviation': 6, 'Undefined': 7}
    df['market_segment'] = df['market_segment'].replace(market_segment_dic)
    # distribution_channel
    distribution_channel_dic = {'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'GDS': 3, 'Undefined': 4}
    df['distribution_channel'] = df['distribution_channel'].replace(distribution_channel_dic)
    # customer_type
    customer_type_dic = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
    df['customer_type'] = df['customer_type'].replace(customer_type_dic)
    # deposit_type
    deposit_type_dic = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}
    df['deposit_type'] = df['deposit_type'].replace(deposit_type_dic)
    # reserved_room_type
    reserved_room_types_dic = {}
    reserved_room_types = df['reserved_room_type'].unique().tolist()
    for x in reserved_room_types:
        reserved_room_types_dic[x] = reserved_room_types.index(x)
    df['reserved_room_type'] = df['reserved_room_type'].replace(reserved_room_types_dic)
    # assigned_room_type
    assigned_room_types_dic = {}
    assigned_room_types = df['assigned_room_type'].unique().tolist()
    for x in assigned_room_types:
        assigned_room_types_dic[x] = assigned_room_types.index(x)
    df['assigned_room_type'] = df['assigned_room_type'].replace(assigned_room_types_dic)
    # year
    for i in range(len(df)):
        df.at[i, 'arrival_date_year'] = int(df.at[i, 'arrival_date_year'] % 100)
        df.at[i, 'adr'] = int(df.at[i, 'adr'] / 10)

    df = df.replace(to_replace=numpy.nan, value=-1)
    df = df.astype({'children': 'int64', 'agent': 'int64', 'company': 'int64', 'country': 'int64', 'adr': 'int64'})

    for x in names:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(x)
        print(df[x].unique().tolist())
        if len(df[x].unique()) == 1:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            df.drop(x, inplace=True, axis=1)

    df = df.drop_duplicates()
    return df


def NN():
    print("~~~~~~~~~~~~~ Neural Network ~~~~~~~~~~~~~~~~")
    mlp = MLPClassifier(hidden_layer_sizes=(4), max_iter=100)
    mlp.fit(X_train, Y_train)
    predictions = mlp.predict(X_test)
    train_eval(predictions)


def RF():
    print("~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~~")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    train_eval(predictions)


def feature_selection(dataframe):
    Y = dataframe['is_canceled'].values
    names.remove('is_canceled')
    X = dataframe[names].values

    # Normalizing the data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = Y.astype('int')

    # feature extraction
    test = SelectKBest(score_func=f_classif, k=10)
    fit = test.fit(X, Y)
    # summarize scores
    set_printoptions(precision=10)
    # print(fit.scores_)
    X_new = fit.transform(X)
    print(X_new.shape)
    # summarize selected features
    mask = test.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, names):
        if bool:
            new_features.append(feature)
    print("selected features :")
    print(new_features)
    return X_new, Y.tolist()


def feature_creation(dataframe):
    dataframe['reservation_status_year'] = pandas.DatetimeIndex(dataframe['reservation_status_date']).year
    dataframe['reservation_status_month'] = pandas.DatetimeIndex(dataframe['reservation_status_date']).month
    dataframe['reservation_status_day'] = pandas.DatetimeIndex(dataframe['reservation_status_date']).day
    dataframe.drop('reservation_status_date', inplace=True, axis=1)
    dataframe['people'] = dataframe['adults'] + dataframe['children'] + dataframe['babies']
    dataframe['not_canceled_minus_canceled'] = dataframe['previous_bookings_not_canceled'] - dataframe['previous_cancellations']
    return dataframe


if __name__ == "__main__":
    dataframe = read_csv("hotel_bookings.csv")
    dataframe = shuffle(dataframe)
    dataframe = feature_creation(dataframe)
    names = list(dataframe.columns.values.tolist())
    dataframe = preprocess(dataframe)
    features, label = feature_selection(dataframe)
    X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=0.3)  # 70% training and 30% test
    NN()
    RF()
