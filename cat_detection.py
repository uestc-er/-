import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import scipy.misc

def load_dataset():
    train_dataset = h5py.File(r'E:\Users\lenovo\AppData\local\Programs\datasets\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(r'E:\Users\lenovo\AppData\local\Programs\datasets\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
# Loading the data (cat/non-cat)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

 # Example of a picture
# 打印出当前的训练标签值
# 使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
# print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
# 只有压缩后的值才能进行解码操作
index = 33
#plt.imshow(train_set_x_orig[index]) 
#plt.show()
#print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")   

# 调用函数加载数据
train_x, train_y, test_x, test_y, classes = load_dataset()

# 查看数据形状（验证加载成功）
print("训练集图像形状：", train_x.shape)  # 输出 (209, 64, 64, 3)
print("训练集标签形状：", train_y.shape)  # 输出 (1, 209)
print("类别列表：", classes)              # 输出 [b'non-cat' b'cat']

m_train = train_set_x_orig.shape[0]   # 训练集里图片的数量。
m_test = test_set_x_orig.shape[0]     # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]    # 训练集里图片的宽度
num_py = train_set_x_orig.shape[2]    # 训练集里图片的宽度


"""
# #看一看 加载的东西的具体情况
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_py) + ", 3)")    
print ("train_set_x shape: " + str(train_set_x_orig.shape))
# test_set_y_orig 为局部变量，返回赋给 train_set_y 了
print ("train_set_y shape: " + str(train_set_y.shape)) 
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
"""



#标准化训练集
#训练集降维转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#测试集降维转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T


"""
#查看降维后维度情况
print('降维后训练集x维度为' , train_set_x_flatten.shape)
print('降维后测试集x维度为' , test_set_x_flatten.shape)
"""


#标准化数据集
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#sigmoid函数（激活函数）
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

"""
#测试sigmoid函数 
print('sigmoid([0,2])=' , sigmoid(np.array([0,2])))
"""

#初始化w和b
def intialize_with_zeros(dim):

    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w , b


"""
#验证一下
dim = 2
w , b = intialize_with_zeros(dim)
print('w = ' + str(w))
print('b = ' + str(b))
"""


#设置a = w^t + b 函数
def propagate(w, b, X, Y):
    """
    实现前向和后向传播的传播函数，计算成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """

    m = X.shape[1]

    #正向传播
    #计算激活函数
    A = sigmoid(np.dot(w.T, X) + b)
    #计算成本
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))         # compute cost

    #反向传播
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    # 创建一个字典，把 dw 和 db 保存起来。 grad 梯度、坡度
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


"""
# 测试一下 propagate 函数
print("====================测试propagate====================")
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
"""


def optimize(w, b, X, Y, num_interations, learning_rate, print_cost = False):  #optimize 优化
    """
    此函数通过运行梯度下降算法来优化w和b
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值
    
    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。
    
    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """

    costs = []
    for i in range(num_interations):

        grads,cost = propagate(w, b, X, Y)    #propagate 宣传、传播

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i % 100 ==0:
            costs.append(cost)
        if print_cost and i % 100 ==0:
            print ("Cost after iteration %i: %f" %(i, cost))
        
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


"""
#测试optimize函数功能
params, grads, costs = optimize(w, b, X, Y, num_interations = 101, learning_rate = 0.009, print_cost = True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print(costs)
"""


def predict(w, b, X):
    """
    使用学习逻辑回归参数 logistic(w，b) 预测标签是0还是1，
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据
    
    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
    
    """
    #图片数量
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    #预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range(A.shape[1]):
        #因为A是在0-1之间的小数，这里将其确定为0或1来量化概率
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0,i] = 1

    # 使用断言
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction


"""
#测试
# 测试一下 predict 函数
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))
"""


#合并模型
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    
    返回：
        d  - 包含有关模型信息的字典。
    """

    #初始化全零系数
    w, b = intialize_with_zeros(X_train.shape[0])

    #参数优化
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)


    # 从“parameters”字典中检索参数w和b
    w = parameters["w"]
    b = parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

print("====================测试model====================")
# 这里加载的是真实的数据
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 5001, learning_rate = 0.001, print_cost = True)           

index = 35
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
plt.show()  # 脚本环境必备，触发图像窗口显示

#绘制损失图像
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()