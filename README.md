# logistic-regression
逻辑回归实战案例
## 1. MNIST Classfification

**参考资料**

1. [MNIST classfification using multinomial logistic + L1](https://scikit-learn.org/dev/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py)

**问题描述**

在这里,我们将逻辑回归与 L1 惩罚以及SAGA 算法应用在 MNIST 数字分类任务上。当样本数明显大于要素数时,这种解算器速度很快,并且能够精细地优化非平滑目标函数,l1 惩罚就是这种情况。测试精度达到 ± 0.8,而权重矢量保持稀疏,因此更易于解释。

**实现代码**

1. mnist_classfification.py
