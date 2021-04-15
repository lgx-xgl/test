import numpy.matlib 
import numpy as np
import pylab as lab
import sklearn
from sklearn import datasets
import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pltfrom 
import random as ran
import matplotlib.pyplot as plt

thi=0.0000032                                                        #thi是梯度下降法的下降步伐,gama是循环次数
gama=90000
class py:                                                           #最小二乘法
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def w_math(self):
        self.x_T=self.x.T
        self.w=np.linalg.inv(np.matmul(self.x_T,self.x))
        self.w=np.matmul((np.matmul(self.w,self.x_T)),self.y)


    def w_out(self):
        return self.w.T


class py_lad:                                                           #梯度下降法
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.count=0
        self.w=np.array(np.int64)
        
    def thita(self,thi,m,zeros):
        while(self.count<gama ):                                    # 循环gama次
           self.w=np.matmul(self.x.T,self.x)
           self.w=np.matmul(self.w,zeros)
           self.w1=np.matmul(self.x.T,self.y)
           self.w=(self.w-self.w1)
           self.w*=thi*2/m                                          #m是数据数量，/m是为了平均数
           self.w=zeros-self.w
           self.count+=1
           if (np.where(self.w!=0)[0].shape[0]==0) : break
           zeros=self.w
        return zeros

def OUT(thita1,thita2):                                             #将图形打印                    
    y1_pre=np.matmul(x_verify,thita1)
    y2_pre=np.matmul(x_verify,thita2)
    plt.xlim([0,50])
    plt.plot(range(len(y_verify)),y_verify,'r',label='y_verify')
    plt.plot(range(len(y1_pre)),y1_pre,'g--',label='pre_ercheng')
    plt.plot(range(len(y2_pre)),y2_pre,'b--',label='pre_tidu')
    plt.legend()
    print("二乘得分:", sm.r2_score(y_verify, y1_pre))
    print("梯度下降得分:", sm.r2_score(y_verify, y2_pre))
    plt.show()

data=datasets.load_boston()
x=data.data
y=data.target
y = np.expand_dims(y, axis=1)

x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)    #分为训练集和测试集
a=x_train.shape[0]
le=x_train.shape[1]

py1=py(x_train,y_train)                                                          #用最小二乘法
py1.w_math()
thita1=py1.w_out()
thita1=thita1.T

py2=py_lad(x_train,y_train)                                                      #用梯度下降法做
zeros=np.matlib.zeros((le))         
zeros=zeros.T                                                                    #起始位置
thita2=py2.thita(thi,a,zeros)

OUT(thita1,thita2)