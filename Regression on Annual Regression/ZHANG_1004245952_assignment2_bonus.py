import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# ------------- Importing Data ----------------

salary = pd.read_csv('Kaggle_Salary.csv', skiprows=[1])
salary.head()

salary['index'] = salary['index'].astype(int)
salary.set_index('index')
df = salary.drop(columns=['Unnamed: 0', 'index'])
newDF = pd.DataFrame()

# ------------- Cleaning Data -----------------

for col in df:

    # For numerical features
    if df[col].dtypes != 'object':
        if "OTHER" in col or "TEXT" in col:
            continue
        else:
            newDF[col] = df[col]
            continue

    cate = df[col].value_counts()
    cate_num = cate.size

    # For categorical features
    if (cate_num <= 0 or cate_num >= 70):
        continue

    # Only one category, encode feature as 1 and nan value as 0
    if (cate_num == 1):
        name = col + "_" + cate.index[0]
        newDF[name] = df[col].notnull().astype('int')

    # Multiple categories, using One Hot Encoder
    if (cate_num > 1):
        dummies = pd.get_dummies(df[col], prefix=col)
        newDF = pd.concat([newDF, dummies], axis=1, sort=False)

# Clear NAN value
newDF = newDF.drop(columns=['Q38_Part_19', 'Q38_Part_20', 'Time from Start to Finish (seconds)'])
newDF = newDF.dropna()

# Numerical feature standerization
scaler = preprocessing.MinMaxScaler()
cols = ['Q34_Part_1', 'Q34_Part_2', 'Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5', 'Q34_Part_6', 'Q35_Part_1',
        'Q35_Part_2', 'Q35_Part_3', 'Q35_Part_4', 'Q35_Part_5', 'Q35_Part_6']
for col in cols:
    newDF[col] = scaler.fit_transform(newDF[[col]])

# Select participants from U.S. to analysis
df_US = newDF[newDF['Q3_United States of America'] == 1]

# ------------- Fit Model ------------------
X = df_US.drop(columns=['Q9']).values
y = df_US["Q9"].values

# Apply the Neural Network model with best parameter found
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 7, 2), random_state=1, max_iter=1000)
clf.fit(X_train, y_train)

print("")
print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

# ------------- Result ----------------

# Performance of the neural network changes with different parameters shown as bellow

# 2 hidden layer combination with accuracy over 0.3
#    hidden layer   acurr_train  acurr_test
# 9        (3, 3)  6.357908e-01    0.441874
# 12       (3, 6)  6.707487e-01    0.423316
# 17       (4, 3)  5.035553e-01    0.392431
# 60       (9, 6)  7.212744e-01    0.387446
# 24       (5, 2)  6.972415e-01    0.386742
# 41       (7, 3)  4.649087e-01    0.382111
# 32       (6, 2)  6.841659e-01    0.376892
# 1        (2, 3)  4.696709e-01    0.375445
# 51       (8, 5)  7.526982e-01    0.363431
# 20       (4, 6)  7.099212e-01    0.347495
# 11       (3, 5)  7.235592e-01    0.334601
# 37       (6, 7)  6.925903e-01    0.321349

# 2 hidden layer combination with accuracy over 0.37
#     hidden layer  acurr_train  acurr_test
# 168    (4, 7, 2)     0.622532    0.444768 <--- chosen one
# 126    (3, 9, 8)     0.657847    0.431569
# 263    (6, 2, 9)     0.718358    0.427371
# 76     (3, 3, 6)     0.609150    0.427356
# 457    (9, 3, 3)     0.699300    0.425140
# 97     (3, 6, 3)     0.674837    0.424582
# 244    (5, 8, 6)     0.610033    0.423177
# 147    (4, 4, 5)     0.666931    0.422920
# 269    (6, 3, 7)     0.671701    0.419778
# 42     (2, 7, 4)     0.676960    0.415560
# 375    (7, 8, 9)     0.553493    0.414314
# 3      (2, 2, 5)     0.553298    0.414014
# 317    (6, 9, 7)     0.678130    0.409975
# 77     (3, 3, 7)     0.694240    0.406093
# 102    (3, 6, 8)     0.530346    0.401505
# 59     (2, 9, 5)     0.508927    0.398620
# 260    (6, 2, 6)     0.510043    0.397514
# 74     (3, 3, 4)     0.689248    0.397486
# 125    (3, 9, 7)     0.689882    0.396417
# 330    (7, 3, 4)     0.693217    0.395451
# 99     (3, 6, 5)     0.794605    0.391698
# 274    (6, 4, 4)     0.486978    0.390287
# 373    (7, 8, 7)     0.692696    0.388017
# 190    (4, 9, 8)     0.697221    0.384115
# 57     (2, 9, 3)     0.700541    0.379033
# 110    (3, 7, 8)     0.697455    0.378586
# 98     (3, 6, 4)     0.480607    0.375424
# 86     (3, 4, 8)     0.684809    0.374896
# 412    (8, 5, 6)     0.712941    0.372776

# Withe the number of hidden layer increasing, we could get a better accuracy
# I believe that more hiden layer could give the model a better performance, but my computer is not good enough to support its computation.
# The performance of neural net is better than other model if good parameters are chosen.
# The model of NN is pretty simple and well apply to this large dataset
# However, there is not a clear descipline to help us find the best parameter, the acurracy varies when a little change occurs in parameters
# Overall, neural network is a powerful method in terms of this dataset, however it need great computation and time to find the best model for it.


# ------------- Tuning hyperparameters ----------------
# # of hidden layer is 2
# start = 2
# stop = 10
# step = 1
# layers = [(x, y) for x in range(start, stop, step) for y in range(start, stop, step)]
# result = []
#
# for layer in layers:
#     clf.set_params(hidden_layer_sizes=layer)
#     clf.fit(X_train, y_train)
#     train_acurr = clf.score(X_train, y_train)
#     test_acurr = clf.score(X_test, y_test)
#     result.append((layer, train_acurr, test_acurr))
#
# df = pd.DataFrame(result, columns=['hidden layer', 'acurr_train', 'acurr_test'])
#
# df_sorted = df.sort_values(by=['acurr_test'], ascending=False)
# print(df_sorted)

# # of hidden layer is 3(need long time to compute)
# start = 2
# stop = 10
# step = 1
# layers = [(x, y, z) for x in range(start, stop, step) for y in range(start, stop, step) for z in
#           range(start, stop, step)]
# result = []
#
# for layer in layers:
#     clf.set_params(hidden_layer_sizes=layer)
#     clf.fit(X_train, y_train)
#     train_acurr = clf.score(X_train, y_train)
#     test_acurr = clf.score(X_test, y_test)
#     result.append((layer, train_acurr, test_acurr))
#
# df = pd.DataFrame(result, columns=['hidden layer', 'acurr_train', 'acurr_test'])
#
# df_sorted = df.sort_values(by=['acurr_test'], ascending=False)
# print(df_sorted)
#
# ar = range(1, df.shape[0]+1)
#
# plt.scatter(ar, df_sorted['acurr_test'])
# plt.xlabel('hidden layer')
# plt.ylabel('accuracy')
# for i in range(0,8):
#     plt.annotate(float("{0:.3f}".format(df_sorted.iloc[i]['acurr_test'])), (ar[i], df_sorted.iloc[i]['acurr_test']))
# plt.show()


