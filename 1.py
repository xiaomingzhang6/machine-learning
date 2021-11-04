# 张晓明
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
dataset = load_boston()
x_data = dataset.data
y_data = dataset.target
name_data =dataset.feature_names
print(name_data)

for i in range(13):
    plt.subplot(7,2,i+1)           #7行2列第i+1个图
    plt.scatter(x_data[:,i],y_data,s=10)  #横纵坐标和点的大小
    plt.title(name_data[i])
    plt.show()
    print(name_data[i],np.corrcoef(x_data[:i]),y_data)
    #打印刻画每个维度特征与房价相关性的协方差矩阵
for i in range(len(y_data)):
    plt.scatter(i, y_data[i], s=10)  # 横纵坐标和点的大小
i_=[]
for i in range(len(y_data)):
    if y_data[i] == 50:
        i_.append(i)#存储房价等于50 的异常值下标
x_data = np.delete(x_data,i_,axis=0)                #删除样本异常值数据
y_data = np.delete(y_data,i_,axis=0)                #删除标签异常值
name_data = dataset.feature_names
j_=[]
for i in range(13):
    if name_data[i] =='RW'or name_data[i] == 'PTRATIO'or name_data[i] == 'LSTAT': #提取'RM'、'PTRATIO'、'LSTAT'三个主要特征
        continue
    j_.append(i)#存储其他次要特征下标
x_data = np.delete(x_data,j_,axis=1)#在总特征中删除次要特征
print(np.shape(y_data))
print(np.shape(x_data))
for i in range(len(y_data)):
    plt.scatter(i,y_data[i],s=10)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(x_data,y_data,random_state = 0,test_size = 0.20)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

#标签归一化的目的是什么呢，实验证明，归一化之后结果好了0.1左右
y_train = min_max_scaler.fit_transform(y_train.reshape(-1,1)) #转化为任意行一列
y_test = min_max_scaler.fit_transform(y_test.reshape(-1,1)) #转化为一列


from sklearn import linear_model
#请完成线性回归的代码，生成lr_y_predict作为测试集的预测结果
lr = linear_model.LinearRegression() #选择线性回归模型
lr.fit(X_train,y_train) #模型的训练
lr_y_predict = lr.predict(X_test) #预测数据

print("lr_y_predict====",lr_y_predict)

from sklearn.metrics import r2_score
score_lr = r2_score(y_test,lr_y_predict)
print("线性回归====",score_lr)


#请完成岭回归的代码，并设置适当的alpha参数值
rr=linear_model.Ridge() #选择模型岭回归
rr.fit(X_train,y_train) #模型的训练
rr_y_predict=rr.predict(X_test)

print('Coefficients: \n', rr.coef_)  #查看相关系数theta
print('Intercept: \n', rr.intercept_)  #查看截距theta0

score_rr = r2_score(y_test,rr_y_predict)
print("岭回归=====",score_rr)

lassr = linear_model.Lasso(alpha=.0001)
lassr.fit(X_train,y_train)
lassr_y_predict=lassr.predict(X_test)

score_lassr = r2_score(y_test,lassr_y_predict)
print("lasso===",score_lassr)

from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1) #高斯核
svr_lin = SVR(kernel='linear', C=100, gamma='auto') #线性核
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1) #径向基核函数

svr_rbf_y_predict=svr_rbf.fit(X_train, y_train).predict(X_test)
score_svr_rbf = r2_score(y_test,svr_rbf_y_predict)
svr_lin_y_predict=svr_lin.fit(X_train, y_train).predict(X_test)
score_svr_lin = r2_score(y_test,svr_lin_y_predict)
svr_poly_y_predict=svr_poly.fit(X_train, y_train).predict(X_test)
score_svr_poly = r2_score(y_test,svr_poly_y_predict)

print("score_svr_poly svr==",score_svr_poly)
print("score_svr_lin svr===",score_svr_lin)
print("score_svr_rbf svr===",score_svr_rbf)

#绘制真实值和预测值对比图
def draw_infer_result(groud_truths,infer_results):
    title='Boston'
    plt.title(title, fontsize=24)
    x = np.arange(-0.2,2)
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results,color='green',label='training cost')
    plt.grid()
    plt.show()


draw_infer_result(y_test,lr_y_predict)
draw_infer_result(y_test,rr_y_predict)
draw_infer_result(y_test,lassr_y_predict)
draw_infer_result(y_test,svr_rbf_y_predict)
draw_infer_result(y_test,svr_lin_y_predict)
draw_infer_result(y_test,svr_poly_y_predict)
print("score of lr:",score_lr)
print("score of rr:",score_rr)
print("score of lassr:",score_lassr)
print("score of svr_rbf:",score_svr_rbf)
print("score of svr_lin:",score_svr_lin)
print("score of svr_poly:",score_svr_poly)
#
#
#
