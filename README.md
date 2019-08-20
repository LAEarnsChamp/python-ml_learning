# ml_learning
用于学习复习
***
聪明人应该用c处理数据
***

##Linear Model
给定由d个属性描述的示例x=(x1, x2, ..., xd),其中xi是x在第i个属性上的取值，线性模型试图学得一个通过属性的线性组合来进行预测的函数
![线性模型](https://upload-images.jianshu.io/upload_images/8199644-077587fee648be49.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
①使损失函数（均方误差）最小，基于这个来进行模型求解的方法称为最小二乘法，它试图找到一条直线，使所有样本到直线上的欧式距离之和最小  
![均方误差最小化](https://upload-images.jianshu.io/upload_images/8199644-98ab5de8f1c593b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
②对w，b求偏导，当导数结果为0时，得到拟合最好的w，b  
![最小二乘法参数估计求偏导](https://upload-images.jianshu.io/upload_images/8199644-065853b4504076b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
《机器学习》将上式取0并对矩阵进行了讨论，从而得到了参数最优解的闭式解。针对更一般的情形（多元线性回归），本代码采用了梯度下降算法拟合参数  
