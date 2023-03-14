import numpy as np
import pandas as pd

#import mnist dataset
data = pd.read_csv("/Users/duongphamminhdung/Documents/GitHub/solving basic mcln problems/handwriting_nums recognize/data/mnist_test.csv", sep=',')
mnist = data.values
label = mnist[:, 0]
digits = mnist[:, 1:]

X_train = digits[:7000]
Y_train = label[:7000]
X_test = digits[-1]
Y_test = label[-1]

def find_target(Y):
    """
    tạo one-hot vector 
    """
    s = np.zeros((len(Y), 10))
    s[np.arange(Y.size), Y] = 1
    return s
def softmax(Z):
    """
    Tỷ lệ rơi vào các class
    Z.shape = (N_class, 1)
    a_i = e^(z_i)/sum(e^Z)
    """
    ez = np.exp(Z-np.max(Z, axis = 1, keepdims = True)) #trừ đi mỗi số giá trị lớn nhất của hàng đó
    A = ez/np.sum(ez) #dãy tỉ lệ rơi vào class 
    return A
def d_softmax(x):
    ez=np.exp(x-x.max())
    return ez/np.sum(ez,axis=0)*(1-ez/np.sum(ez,axis=0))
def activate_func(Z):
    #activate
    Z[Z <= 0] = 0
    return Z
def calc(X, Y, W1, W2):
    """ 
    X, Y : 128x784
    W1: 784x128
    W2: 128x10
    """
    targets = find_target(Y)        #target     128x10
    x_1=X.dot(W1)                   #x_1        128x128
    x_activate=activate_func(x_1)   #x_activate  128x128
    x_2=(x_activate).dot(W2)        #x_2        128x10
    cost=softmax(x_2)               #cost       128x10

    loss = (np.sum(-targets*np.log(cost), axis = 1)/len(y))[..., None] #128, 1
    update_2=x_activate.T@loss


    loss=((W2).dot(loss.T)).T*(x_1)
    update_1=X.T@loss

    return cost,update_1,update_2 

W1 = np.random.randn(784, 128)
W2 = np.random.randn(128, 10)

epoch = 1000
acc = []
losses = []
lr = 0.005
batch = 128

for i in range(epoch):
    #randomize and create batches
    sample=np.random.randint(0,X_train.shape[0],size=(batch))
    x=X_train[sample]
    y=Y_train[sample][:, None]

    out,update_1,update_2= calc(x, y, W1, W2)   
    category=np.argmax(out,axis=1)
    
    accuracy=(category==y).mean()
    acc.append(accuracy.item())
    
    loss=((category-y)**2).mean()
    losses.append(loss.item())
    
    #SGD 
    W1=W1-lr*update_1
    W2=W2-lr*update_2

    if i%100 == 0:
        print("epoch", i, "accuracy:", acc[-1])
test=np.argmax(softmax((X_test.dot(W1)).dot(W2)),axis=1)
accuracy=(test==Y_test).mean().item()
print(f'Test accuracy = {accuracy:.4f}')
