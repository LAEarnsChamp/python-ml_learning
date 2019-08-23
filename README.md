# ml_learning
用于学习复习
***
聪明人应该用c处理数据
***

## Linear Model
给定由d个属性描述的示例x=(x1, x2, ..., xd),其中xi是x在第i个属性上的取值，线性模型试图学得一个通过属性的线性组合来进行预测的函数  
![线性模型](https://upload-images.jianshu.io/upload_images/8199644-077587fee648be49.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
①使损失函数（均方误差）最小，基于这个来进行模型求解的方法称为最小二乘法，它试图找到一条直线，使所有样本到直线上的欧式距离之和最小  
![均方误差最小化](https://upload-images.jianshu.io/upload_images/8199644-98ab5de8f1c593b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
②对w，b求偏导，当导数结果为0时，得到拟合最好的w，b  
![最小二乘法参数估计求偏导](https://upload-images.jianshu.io/upload_images/8199644-065853b4504076b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
《机器学习》将上式取0并对矩阵进行了讨论，从而得到了参数最优解的闭式解。针对更一般的情形（多元线性回归），本代码采用了梯度下降算法拟合参数  

## Linear Discriminant Analysis
![LDA示意图](https://upload-images.jianshu.io/upload_images/8199644-7d8b8e4dfb644637.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
μi表示第i类点的均值向量，Σi表示第i类点的协方差矩阵，则:  
![广义瑞丽商](https://upload-images.jianshu.io/upload_images/8199644-b726da47f6cffd15.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
超平面w：  
![超平面w](https://upload-images.jianshu.io/upload_images/8199644-bc9f426333cea19b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
《机器学习》讲得实在晦涩，实现代码参考了  
https://www.jianshu.com/p/28da5c160230  
以及  
https://blog.csdn.net/weixin_40604987/article/details/79615968  
在此基础上本代码结合《机器学习》中的推导式给出了投影超平面w、特征值及对应的特征向量（矩阵形式给出）  