import numpy as np
import pandas as pd

#import mnist dataset
data = pd.read_csv("/Users/duongphamminhdung/Documents/GitHub/DATA SETS/mnist/mnist_test.csv", sep=',')
mnist = data.values
label = mnist[:, 0]
digits = mnist[:, 1:]

X_train = digits[:7000]
Y_train = label[:7000]
X_test = digits[-10:]
Y_test = label[-10:]
def normalize(X):
    X = X / 255
    return X
def find_target(Y):
    """
    tạo one-hot vector 
    """
    Y = Y.reshape(len(Y))
    s = np.zeros((len(Y), 10))
    s[np.arange(Y.size), Y] = 1
    return s
def softmax(Z):
    """
    Tỷ lệ rơi vào các class
    Z.shape = (N_class, 1)
    a_i = e^(z_i)/sum(e^Z)
    """
    ez = np.exp(Z) 
    A = ez / np.sum(ez, axis = 1)[:, None] #dãy tỉ lệ rơi vào class 
    return A
def d_softmax(x):
    ez=np.exp(x-x.max())
    return ez/np.sum(ez,axis=0)*(1-ez/np.sum(ez,axis=0))
def activate_func(Z):
    #activate
    Z[Z <= 0] = 0
    return Z
def calc(x, y, W1, W2):
    """ 
    X, Y : 128x784
    W1: 784x128
    W2: 128x10
    """
    targets = find_target(y)        #target     batchsizex10
    x_1=x.dot(W1)                   #x_1        batchsizex128
    x_activate=activate_func(x_1)   #x_activate batchsizex128
    x_2=(x_activate).dot(W2)        #x_2        batchsizex10
    cost=softmax(x_2)               #cost       batchsizex10
    loss = (np.mean(-targets*np.log(cost))) #batchsize, ()
    update_1=x.T.dot(loss)                  #update1    784xbatchsize
    update_2=x_2.T.dot(loss )               #update2    10xbatchsize
    
      
    return cost, loss, update_1, update_2 

epoch = 1000
acc = []
losses = []
lr = 0.05
batch = 100
i = 0
W1 = np.random.randn(784, batch)
W2 = np.random.randn(batch, 10)

while i < epoch:
    i += 1
    sample=np.random.randint(0,X_train.shape[0],size=(batch))
    x=X_train[sample]
    y=Y_train[sample][:, None]
    x = normalize(x)
    out,loss, update_1,update_2= calc(x, y, W1, W2)   
    category=np.argmax(out,axis=1)
    
    accuracy=(category==y).mean()
    acc.append(accuracy.item())
    
    losses.append(loss)
    
    #SGD 
    W1=W1-lr*update_1
    W2=W2-lr*update_2.T

    if i%100 == 0:
        print("epoch", i, "accuracy:", acc[-1])
print(losses)
# test = softmax((activate_func(X_test.dot(W1))).dot(W2))
# accuracy=(test==Y_test).mean().item()
# print(f'Test accuracy = {accuracy:.4f}')
