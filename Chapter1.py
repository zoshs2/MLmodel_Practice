## Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv())
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # ignore warings like DeprecationWarning(비권장사항 경고)
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "./inputs"]).decode("utf8"))

data = pd.read_csv("./inputs/column_2C_weka.csv")
print(data.head())
print(data.describe())

## MACHINE LEARNING (ML)
# In python there are some ML libraries like sklearn, keras or tensorflow.
# We will use sklearn.

# A. Supervised Learning
# : It uses data that has 'labels'. Example, there are orthopedic patients(정형외과 환자)
# data that have labels 'normal' & 'abnormal'.
# - There are "Features(:predictor variable;예측변수)" & "Target variable(타겟변수)".
#   'Features' are like 'pelvic(골반의) radius' or 'sacral slope'.
#   'Target variables' are labels normal and abnormal.
# (예측변수(predictor variable)은 즉 입력변수를 말한다.)
# (타겟변수(target variable)은 즉 레이블을 말한다.)
# - Aim is that as given features(input) predict whether target variable(output)
#   is 'normal' | 'abnormal'.
# - Classification : target variable consists of categories like normal | abnormal.
# - Regression : target variable is continuous like stock market.
# If these explanations are not enough for you, just google them.
# However, be careful about terminology :
# -> Features = predictor variable = independent variable = columns = inputs.
# -> Target variable = labels = response variable = class = dependent variable = ouput = result


## Exploratory Data Analysis (EDA) : 탐색적 데이터 분석
# 데이터에 대한 기본적인 직관과 이해를 위한 기초적인 시각화 기법 또는 통계조사
# - In order to make something in data, as you know you need to  explore ddata.
#   Detailed exploratory data analysis is in my Data Science Tutorial for Beginners.
# - I always start with head() to see features that are pelvic_incidence, pelvic_tiltnumeric, 
#   lumbar_lordosis_angle, sacral_slope, pelvic_radius and degree_spondylolisthesis and 
#   target variable that is class.

# to see feautures and target variable
print(data.head())

# Want to know if there is any NaN value and length of this data so lets look at info.
print(data.info())
# -> 310개의 엔트리들 중 nan value는 없음을 알 수 있다.

'''
color_list = ['red' if cls=='Abnormal' else 'green' for cls in data.loc[:,'class']]
# column들중 class라는 열을 제외하고 모두 = (data.loc[:, data.columns != 'class'])
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'], c=color_list, figsize=[15,15], diagonal='hist', alpha=0.5, s=200, marker='*', edgecolor='black')
# pd.plotting.scatter_matrix(edgecolor='??') -> 히스토그램이나 마커들의 테두리 색을 말한다. 
plt.show()
'''

# Okay, as you understand in scatter matrix there are relations between each feauture.
# But, you don't know yet how many normal(green) & abnormal(red) classes are there.
# - seaborn(sns) library has countplot() that counts number of classes. (시각적으로 보여줌)
# - Also, you can print it with .value_counts() method of pandas(pd). (수치적으로 보여줌)
'''
print(sns.countplot(x="class", data=data))
print(data.loc[:,'class'].value_counts())
# -> Abnormal : 210 & normal : 100
'''

## K-Nearest Neighbors (KNN) : Classification method.
# 현실에서는 특정한 확률 분포를 따르지 않는 경우가 매우 많다.
# 이러한 특정되지 않는 확률분포를 추정하기 위해 사용하는 방법을 '비모수적 방법'(non-parametric method)
# 라고 하는데, 대표적으로 '파젠 창(Parzen Window)'과 '최근접이웃(KNN)'방법이 있다.
# 그렇지만 사실 따지고 보면, KNN은 확률분포추정을 위한 방법이 아니라 
# '분류(classification)'를 위한 방법이다. 하지만 동작원리적 측면에서 비모수적 방법으로 소개한다.

# Parzen Window 방법은 고정된 크기(h)의 창을 이용하여 창의 중심 x를 어디에 두느냐에 따라 창 안의 표본 갯수가 달라진다.
# 하지만 이와 반대로 작동하는 방법을 고안할 수 있다.
# 이 새로운 방법에서는 x를 중심으로 창을 씌우고, k개의 표본이 창 안에 들어올 때까지 창의 크기를 확장해 나간다.
# k개가 들어온 순간의 창의 크기를 h라고 한다. k는 고정되어 있고 h가 중심 x에 따라 변하는 것이다.
# (파젠 창에서는 창의 크기 h가 고정되고 k가 x에 따라 변하였다.)
# 직관적으로 생각해보면, 큰 h값을 가진 x주위에는 포본이 희소하게 분포함을 뜻하므로 그곳에서는 확률이 낮아야 하며
# 반대로 h가 작은 x는 높은 확률을 가져야 한다.
# 이 원리를 바탕으로 p(x) = (1/(h_x)^d) * (k / N)를 이용하여 확률분포를 추정할 수 있다.
# 이 방법을 k-최근접 이웃 추정(k-nearest neighbors estimate)이라고 한다.
# 이 작업의 시간 복잡도는 Theta(kdN)이다. 
# -> Theta(x)는 어느 경우든지 x만큼 수행한다는 뜻이다.
#    반면 일반적으로 아는 O(x)는 기껏해야 x만큼 한다는 뜻이다. (즉 상한을 뜻함)
#    Omega(x)는 적어도 x만큼 한다는 뜻이다. (즉 하한을 뜻함)
# 즉 K-nearest neighbors estimate는 훈련 집합의 크기(k)이 크고, 공간의 차원(d)이 높을 때 계산량이 무척 많다.
# 이를 극복하고 계산속도를 빠르게 하기 위해, 훈련 집합에 따라 특징 공간을 미리 여러 구간으로 나누어 놓는
# 보르노이 도형(Voronoi diagram)방법을 활용할 수 있다.

# KNN 분류기(classifier)는 위의 원리를 바탕으로 한다.
# 알고리즘 :
# 1. 훈련샘플 중에 x에 가장 가까운 k개 샘플들을 찾는다.
# 2. k가 속한 부류를 조사하여 가장 빈도가 높은 부류로 x를 분류한다.

# First we need to train our data. (Train corresponds .fit method.)
# .fit() : fits the data, train the data.
# predict() : predicts the data.
# x : features.
# y : target variables(normal | abnormal)

# KNN (여기서는 3-NN)을 쓴다.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # 3-NN


# Today will be rested.
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
print("type(x) : ", type(x), "\n and type(y) : ", type(y))
# x : <class 'pandas.core.frame.DataFrame'>
# y : <class 'pandas.core.series.Series'>
knn.fit(x,y) # train the knn model.
prediction = knn.predict(x)
print("Prediction: {}".format(prediction))

# Well, we fit(train) data(model) and predict it with KNN.
# So, do we predict correct or what is our accuracy or the accuracy is the best metric to 
# evaluate our result? Lets give answer of this questions.
# Measuring model performance :
# Accuracy which is the fraction of correct predictions is commonly used metric.
# We will use it know but there is another problem.

# As you see I train data with x(features) and again predict the x(features). Yes 
# you are reading right but yes you are right again it is absurd :)
# -> 즉, 트레이닝 한 데이터로 predict한다는게 말이 안된다는 말.

# Therefore we need to split our data train and test sets.
# 트레이닝 전에 train data와 test data 을 나눠줘야 한다.

# Train & Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
# x,y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("With KNN(K=3) accuracy is : ", knn.score(x_test, y_test)) # Accuracy
# Accuracy is 86% so is it good?

# Now the question is why we choose K=3 or what value we need to choose K.
# (K는 몇이 적당할까? 어떻게 결정할까?)
# The answer is in 'Model Complexity'. 
# -> K has general name. It is called a hyper-parameter.
# For now just know K is hyper-parameter and we need to choose it that
# give the best performance.

# -> If K is small, model is complex model can lead to "Overfit"!!
# It means that model memorizes the train sets and cannot predict test set with good accuracy.

# -> If K is big, model is less complex model can lead to "Underfit"!!

# At below, I range K value from 1 to 25 and find accuracy for each K value.
# As you can see in plot, when K is 1, it memorizes the train sets (=Overfit)
# and cannot give good accuracy on test set. 
# Also if K is 18, model is lead to "Underfit". 
# Again accuracy is not enough. However look at when K is 18(best performance),
# accuracy has highest value almost 88%
neig = np.arange(1,25)
train_accuracy = []
test_accuracy = []
# Loop over different values of K
for i, k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=(13,8))
plt.plot(neig, test_accuracy, label="Testing Accuracy")
plt.plot(neig, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Performance Comparing")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neig)
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))
