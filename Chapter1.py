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
'''
plt.figure(figsize=(13,8))
plt.plot(neig, test_accuracy, label="Testing Accuracy")
plt.plot(neig, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Performance Comparing")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neig)
plt.show()
'''
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))

## REGRESSION (Linear & Logistic Regression)
# This orthopedic patients(정형외과 환자) data is not proper for regression 
# so I only use two features are 'sacral_slope' & 'pelvic_incidence' of abnormal.
# -> I consider feature is 'pelvic_incidence' and target is 'sacral_slope'.
# -> Lets look at scatter plot so as to understand it better.
# -> reshape(-1,1): If you do not use it shape of x or y becames (210,)
#    and we cannot use it in sklearn, so we use shape(-1,1)
#    and shape of x or y be (210, 1).

# create data1 that includes 'pelvic_incidence' that is feature(input).
# and 'sacral_slope' that is target variable(output; label).
data1 = data[data['class']=='Abnormal'] # DataFrame
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
# Scatter
'''
plt.figure(figsize=(10,10))
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
'''
# Now we have our data to make regression.
# In regression problems, target value is continuously varying variable such as
# price of house or sacral_slope. 
# Lets fit line into this points...!!!

# Linear Regression
# y = ax + b where y is target's value, x = feature's value and "a = parameter of model".
# 여기서 a가 train되는 것이다.!!!!
# We choose parameter of model(a) according to minimum error function that is loss function.
# -> loss function 을 가장 작게 만드는 model의 파라미터 a를 설정해야한다.
# In Linear Regression we use Ordinary Least Square(OLS) as loss function.
# OLS : sum all residuals but some positive and negative residuals can cancel each other
#       so we sum of square of residuals. It is called OLS.
# Score : Score uses R^2 method that is ((y_pred - y_mean)^2) / (y_actual - y_mean)^2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# Predict Space
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
# Fit(Train)
reg.fit(x,y)
# Predict
predicted = reg.predict(predict_space)
# R^2 method = ((y_pred - y_mean)^2) / ((y_actual - y_mean)^2)
print("R^2 score : ", reg.score(x, y))
# Plot regression line and scatter
'''
plt.plot(predict_space, predicted, color="black", linewidth=3)
plt.scatter(x=x, y=y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()
'''

## Cross Validation
# In KNN method before, we use train_test_split with random_state that split exactly
# same at each time. (random_state는 일종의 seed로서 매번 똑같은 분포로 나뉘어진다.)
# However, if we do not use random_state, data is split differently at each time and
# according to split accuracy will be different.
# Therefore, we can conclude that model performance is 'dependent' on train_test_split.
# For example you split, fit and predict data 5 times and
# accuracies are 0.89, 0.9, 0.91, 0.92 and 0.93, respectively.
# Which accuracy do you use? Do you know what accuracy will be at 6th times split, train and predict.
# The answer is I don't know but if I use 'Cross Validation(CV)' I can find acceptable accuracy.

# K folds = K fold CV
# When K is increase, computationally cost is increasing.
# cross_val_score(reg, x, y, cv=5) : use reg(Linear Regression) with x and y
# that we define at above and K is 5. It means 5 times(split, train, predict)
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score
print("CV Scores : ", cv_result) 
print("CV scores average : ", np.sum(cv_result)/k)
# -> reg.fit 즉, 아직 학습을 안시켜서 score가 낮음.

## Regularized Regression
# As we learn linear regression choose parameters(=coefficients) while minimizing lost function.
# If linear regression thinks that one of the feature is important,
# it gives 'high coefficient' to this feature. 
# However, this can cause 'OVERFITTING PROBLEM' that is like memorizing in KNN.
# In order to avoid overfitting., we use "Reguralization" that penalize large coefficients.

# - Ridge(산등성이) regression : First regularization technique. 
# Also, it is called "L2 Regularization".
#   - Ridge regression lost function = OLS(sum of square of residuals) + (alpha * sum(absolute_value(parameter)^2))
#   - alpha is (hyper-)parameter we need to choose to fit(train) and predict.
#   - Picking alpha is similar to picking K in KNN. 
#     (* 우리는 KNN에서 K를 고를 때, 1~25부터 쫙 결과를 나열해보고 K=18일때 가장 퍼포먼스가 좋았다.)
#   - So alpha is hyperparameter that we need to choose for the best accuracy and model complexity.
#     이러한 최적의 hyper-parameter값을 찾아내는 과정(process)을 "Hyperparameter Tuning"이라고 한다.
#   - alpha 값이 0이라면, Ridge regression lost function은 그냥 OLS가 되며 이는 Linear Regression와 같다.
#   - If alpha is small that can cause OVERFITTING. (while being big, it can cause UNDERFITTING)
#   - Do not as what is small and big. These can be changed from problem to problem. (케바케라는 뜻)

# - Lasso(올가미밧줄) regression : Second regularization technique. 
# Also, it is called "L1 Regularization".
#   - Lasso regression lost function = OLS(sum of square of residuals) + (alpha * sum(absolute_value(parameter)^1))

# Linear vs. Ridge(L2) vs. Lasso(L1)

# Ridge(L2) regression
from sklearn.linear_model import Ridge
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2, test_size=0.3)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(x_train, y_train)
ridge_predict = ridge.predict(x_test)
print("Ridge score : ", ridge.score(x_test, y_test))

# Lasso(L1) regression
from sklearn.linear_model import Lasso
x = np.array(data1.loc[:, ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'pelvic_radius']])
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=3, test_size=0.3)
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x_train, y_train)
ridge_predict = lasso.predict(x_test)
print('Lasso score : ', lasso.score(x_test, y_test))
print("Lasso coefficients : ", lasso.coef_) # Oh
# lasso.coef_ = [0.82498243, -0.7209057, 0., -0.]
# -> As you can see 'pelvic_incidence' & 'pelvic_tilt numeric' are 'Important Features' 
#    but others are not important.

# We need to use "Confusion Matrix" as model measurement matrix in imbalance data.
# While using Confusion Matrix, lets use Random Forest Classifier to diversify classification methods.
# tp : Prediction is positive(normal) & Actual is positive(normal). (빈도 수)
# fp : Prediction is positive(normal) & Actual is negative(abnormal).
# tn : Prediction is negative(abnormal) & Actual is negative(abnormal).
# fn : Prediction is negative(abnormal) & Actual is positive(normal).
# Precision = tp / (tp+fp) : 정확도; Positive(p)로 예측한 관측치 중 실제 Positive한 관측치 비율을 나타내는 지표.
# Recall = tp / (tp+fn) : 재현율; 실제값이 Positive한 관측치 중 적중한 예측치의 비율을 나타내는 지표.
# f1 (score)= ( 2 * Precision * Recall ) / (Precision + Recall) (f_beta지표도 있다.)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1, test_size=0.3)
rf = RandomForestClassifier(random_state=4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix : \n", cm)
print("Classification Report : \n", classification_report(y_test, y_pred))
# RF랑 Confusion matrix 공부하기.

# Visualize with seaborn library
'''
sns.heatmap(cm, annot=True, fmt="d")
plt.show()
'''

## ROC(Receiver Operating Characteristic) Curve with Logistic Regression
# 레이더 이미지 분석 성능을 평가하기 위해 개발된 ROC Curve.
# -> 두 분류 분석 모형을 비교 분석 결과를 가시화할 수 있다는 점에서 유용한 평가 도구이다.
# (즉, 오분류표(Confusion Matrix)는 하나의 분류분석모형을 집중 평가한데 반해, 
#      ROC Curve는 두 가지 모형을 비교하여 우위를 가린다.)
# * Sensitivity(민감도) 와 Specificity(특이도)에 대해 알아야 한다.
# 민감도와 특이도의 개념은 오분류표(Confusion Matrix)에서부터 시작한다.
# - Sensitivity(민감도)
#   = TP / P : Positive한 모든 실제 관측치 중 예측치가 적중한 정도.
# - Specificity(특이도)
#   = TN / N : Negative한 모든 실제 관측치 중 예측치가 적중한 정도.
# 모형의 성과를 평가하는 기준은 ROC Curve의 밑부분 면적(Area Under the ROC Curve, AUC)이 넓을 수록 
# 좋은 모형으로 평가한다. -> AUC가 1에 가까울수록 좋은 모형으로 평가된다.
# (x축 : FP Ratio(1-특이도), y축 : 민감도(Sensitivity))
# -> 즉 ROC그래프의 x축은 FP / N을 말한다. (False Positive rate)
#      y축은 TP / P 로 True Positive rate이다.
# sklearn.metrics의 roc_curve메소드로 쉽게 구현할 수 있다.

# Logistic Regression output is probabilities. (엄밀히 말하면, 이 확률을 사후확률(Posterior Probability)라고 한다.)
# 정리하자면, 예측변수 값이 주어질 때, 반응변수(target variable; label;)의 각 범주에 속할 확률이 얼마인지 추정하며(여기까진 예측모형)
# 이 추정확률(사후확률)을 기준치에 따라 분류하는 목적으로 사용하는 것이다. (여기까지가 분류모형)
# 따라서 로지스틱 회귀모형은 범주형 반응변수에 적용되는 회귀분석 모형이다.
# If probability is higher than 0.5 data is labeled 1(abnormal) else 0(normal).
# By default logistic regression threshold is 0.5
# In this curve, x axis is False Positive rate(FP/N) & y axis is True Positive rate(TP/P)
# If the curve in plot is closer to left-top corner, model is more accurate.
# fpr : False Positive rate (FP/N)
# tpr : True Positive rate (TP/P)

from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀(범주형)
from sklearn.metrics import confusion_matrix, classification_report
# abnormal = 1 & normal = 0
data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]
x, y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:, 'class_binary']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(x_train, y_train) # training
 # LogisticRegression에는 .predict_proba()라는 범주별 확률을 반환하는 메소드가 있다.
y_pred_prob = logreg.predict_proba(x_test)[:,1]
# shape of (93, 2) : 첫 번째 열은 0-클래스에 속할 확률이고, 두 번째 열은 1-클래스에 속할 확률이다.
# 따라서 두 확률의 합은 항상 1이다.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC Curve
'''
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate(FP/N)")
plt.ylabel("True Positive Rate(TP/P)")
plt.title('ROC Curve')
plt.show()
'''

