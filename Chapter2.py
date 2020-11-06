## Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv())
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings # ignore warings like DeprecationWarning(비권장사항 경고)
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "./inputs"]).decode("utf8"))

data = pd.read_csv("./inputs/column_2C_weka.csv")

# B. Un-Supervised Learning (비지도 학습)
# Unsupervised Learning : It uses data that has 'unlabeled' and 'uncover' hidden 
# patterns from unlabeled data. For example, if there are orthopedic patients(성형 환자) data
# that do not have labels. You do not know which orthopedic patient is normal or abnormal.

# But as you know, the orthopedic patients data is labeled(normal or abnormal) data.
# In other words, it has target variables(=labels). 
# So, in order to work on 'Unsupervised learning', lets drop target variables and 
# to visualize just consider 'pelvic_radius' and 'degree_spondylolistthesis' attributes.

# -> 정리하자면, 비지도 학습은 라벨이 없는 데이터를 다루고, 그런 라벨이 없는 데이터로부터 숨겨진 패턴을 찾아내는 일을 수행한다.
# 여기서는 orthopedic patients데이터를 다루는 데, 사실 이 데이터에는 라벨이 있다. (normal과 abnormal)
# 비지도 학습의 컨셉에 맞게끔, 이 라벨을 떼어내고 진행할 것이다.

## KMeans (Cluster)
# First unsupervised learning is 'KMeans Cluster'.
# KMeans Cluster : 
# 각 군집은 하나의 중심(centroid)을 가진다. 그리고 각 개체는 가장 가까운 중심(centroid)에 할당된다. 
# 그리고 같은 중심에 할당된 개체들이 모여 '하나의 군집'을 형성한다.
# 이 때 군집 수(k)는 사용자가 사전에 정해야 알고리즘을 실행할 수 있으며, 따라서 군집 수(k)는 hyper-parameter이다.
# KMeans Cluster는 "EM알고리즘"을 기반으로 하는데.
# EM알고리즘은 Expectation과 Maximization 스텝(step)으로 나뉜다.

# 예컨대, 초기 군집 수(k)를 2로 정했으면, 그 군집의 중심(centroid)를 랜덤초기화로 할당한다.
# 그럼 나머지 모든 개체들이 가장 가까운 중심들에 각각 할당된다. 이 과정이 'Expectation'스텝이다.
# 그렇게 할당되고나면, 기존 중심과 할당된 개체들 사이의 중심으로 중심(centroid)를 다시 업데이트한다. 이 과정이 'Maximization'스텝이다.
# 그리고 업데이트된 각각의 중심을 기준으로 각 개체들을 가장 가까운 중심에 맞게 재할당한다. (Expectation스텝)
# 재할당이 끝나면, 또 다시 중심을 업데이트하는 Maximization스텝을 수행한다.
# 이런 과정을 반복하며, 그 결과가 변하지 않거나(즉, 해가 수렴) 또는 사용자가 정한 반복 횟수를 채우면 학습이 끝이 난다.
# 이 중심점들은 결국 각 군집의 데이타의 평균값을 위치로 가지게 되는데, 이런 이유로 Means(평균) 값 알고리즘이라고 한다.


# os.system("pwd")
data = pd.read_csv("/Users/yungi/Documents/Hello_Atom/MLmodel_Practice/inputs/column_2C_weka.csv")
# As you can see there is no labels in data. 
class_data = data['class']
data.drop('class', axis=1, inplace=True)
# print(data.columns)
'''
plt.scatter(data['pelvic_radius'], data['degree_spondylolisthesis'])
plt.xlabel("pelvic_radius")
plt.ylabel("degree_spondylolisthesis")
plt.show()
'''

data2 = data.loc[:, ['degree_spondylolisthesis', 'pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2) # 군집 수 k=2
kmeans.fit(data2)
labels = kmeans.predict(data2) # 군집 수만큼 0~(k-1)로 라벨을 매긴 결과를 반환한다.
'''
plt.scatter(data['pelvic_radius'], data['degree_spondylolisthesis'], c=labels)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
'''
# 그림을 통해, (색깔로) 두 개의 군집으로 나눠진 것을 알 수가 있다.

## Evaluating of Clustering (군집모델 평가하기)
# We cluster data in two groups. Okay well is that correct clustering?
# In order to 'evaluate' the clustering, we will use 'cross tabulation' table.
# (Cross Tabulation table -> https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221155659188&proxyReferer=https:%2F%2Fwww.google.com%2F)
# - There are two clusters that are 0 and 1.
# - First class 0 includes 138 abnormal & 100 normal patients.
# - Second class 1 includes 72 abnormal & 0 normal patients.
#   * The majority of two clusters are abnormal patients.

# Cross Tabulation Table
'''
df = pd.DataFrame({'label':labels, 'class':class_data})
ct = pd.crosstab(df['label'],df['class'])
print(ct)
'''

# 적정 군집 수(k)를 파악하는 방법
# KMeans를 수행하기 전에는 클러스터의 개수(k)를 명시적으로 지정해줘야 한다.
# 데이터를 2개로 군집화(clustering)할 것인지, 3개로 할 것인지 등..
# 그렇다면, 몇 개의 클러스터 수(k)가 가장 적절한지 어떻게 알 수 있을까?
# "Inertia value"라는 값을 보면, 적정 클러스터 수를 선택할 수 있는 힌트를 얻을 수 있다.
# Inertia value : 
# 군집화가 된 후에, 각 군집별 중심점(centroid)으로부터 군집 내 데이터들 사이의 거리를 합산(sum)한 것으로
# 일종의 '군집의 응집도'를 나타내는 값이다.
# 따라서 이 inertia value값이 작을 수록 '군집의 응집도가 높다'는 뜻이고, 잘 군집화되었다고 볼 수 있다.
# 하지만 그러면서도, 일반적으로 inertia value가 낮을 수록, 클러스터의 수(k)도 많아지는 trade off 경향이 있으므로
# 과하게 클러스터 수가 너무 많지는 않으면서도 inertia value가 작은 적당선(elbow)를 캐치하는 것이 포인트다!

# Inertia Value
inertia_list = np.empty(8)
for i in range(0,8):
    kmeans = KMeans(n_clusters=i+1)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_

'''
plt.plot(range(1,9), inertia_list, '-o')
plt.xlabel("Number of Cluster (k)")
plt.ylabel("Inertia Value")
plt.show()
'''
# 그래프를 봤을 때, k=3~5 사이가 적당한 선택일 것 같다.

## Standardization (표준화)
# 학습 데이터의 각 속성별 값의 범위가 넓고, 그 범위가 다른 속성값과 차이도 크다면, 머신러닝 학습이 잘 안되는 경우가 있다.
# 예를 들어, 속성 A의 값 범위가 1~10000 사이 정도이고, 속성 B의 값 범위가 1~10 사이라면
# 학습이 제대로 되지 않을 수 있다.
# 그래서 각 속성의 값 범위를 (동일하게) 일치시켜주는 과정을 "스케일링(Feature Scaling)"이라고 한다.
# 이 스케일링에 대한 여러가지 기법과 알고리즘이 있다.
# 여기서는 속성들의 모든 값을 0~1사이의 값으로 일치시켜주는 StandardScaling방법을 사용한다.
# -> from sklearn.preprocessing import StandardScaler ( * Do not forget standardization as pre-processing. )
# 즉, 학습을 하기 전에 먼저 데이터를 StandardScaler로 스케일링해주고, 이렇게 스케일링된 데이터를 
# 모델에 넣어서 학습시킨다. 
# 이는 지도학습와 비지도학습에서 모두 중요한 절차이다.
# We can use pipeline.
# Pipeline : The purpose of the pipeline is to assemble several steps 
# like KMeans(cluster) and StandardScaler(pre-processing). 
# (즉, 일련의 절차를 하나로 묶는 것. 전처리와 모델학습을 묶음; Pipeline)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline # 파이프라인! 이걸로 전처리과정과 모델학습과정을 합칠거다.
scaler = StandardScaler()
kmeans = KMeans(n_clusters=2) # 클러스터 수(k) = 2
pipe = make_pipeline(scaler, kmeans) # 스케일링과정(StandardScaler)과 모델학습(Kmeans)과정을 파이프라인으로 묶음.
pipe.fit(data)
labels = pipe.predict(data) # kmeans는 군집화 후, 그 결과를 0~(k-1)사이의 정수 값으로 라벨링하여 반환한다.
df = pd.DataFrame({'labels':labels, 'class':class_data})
ct = pd.crosstab(df['labels'], df['class']) # Cross Tabulation Table
print(ct)




