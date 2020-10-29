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

print(sns.countplot(x="class", data=data))
print(data.loc[:,'class'].value_counts())
# -> Abnormal : 210 & normal : 100

## K-Nearest Neighbors (KNN)
# 현실에서는 특정한 확률 분포를 따르지 않는 경우가 매우 많다.
# 이러한 특정되지 않는 확률분포를 추정하기 위해 사용하는 방법을 '비모수적 방법'(non-parametric method)
# 라고 하는데, 대표적으로 '파젠 창(Parzen Window)'과 '최근접이웃(KNN)'방법이 있다.
# 그렇지만 사실 따지고 보면, KNN은 확률분포추정을 위한 방법이 아니라 
# '분류(classification)'를 위한 방법이다. 하지만 동작원리적 측면에서 비모수적 방법으로 소개한다.

