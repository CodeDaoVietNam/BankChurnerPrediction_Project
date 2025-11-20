import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#==========================
# LOGISTICREGRESSION MODEL=
#==========================
class MyLogisticRegression:
    def __init__(self,lr = 0.1,epochs= 1000,batch_size = None ,l2 = 0.0,verbose = False):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def initialize_weights(self,n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def compute_loss(self,y_pred,y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred,eps,1-eps)
        ce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) #Công thức tính toán hàm mất mát
        if self.l2 > 0:
            ce += self.l2 * np.sum(self.w**2)
        return ce

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        if self.batch_size is None:
            self.batch_size = n_samples

        for epoch in range(self.epochs):
            shuffle_idx = np.random.permutation(n_samples)
            X = X[shuffle_idx]
            y = y[shuffle_idx]

            #Mini_batch
            for start in range(0,n_samples,self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                z = np.dot(X_batch,self.w) + self.b
                y_pred = self.sigmoid(z)
                dw = np.dot(X_batch.T,(y_pred-y_batch))/len(X_batch)
                db = np.mean(y_pred-y_batch)

                if self.l2 > 0:
                    dw += self.l2*self.w
                self.w -= self.lr*dw
                self.b -= self.lr*db

            #Compute loss each epoch
            z_all = np.dot(X,self.w) + self.b
            y_pred_all =self.sigmoid(z_all)
            loss = self.compute_loss(y_pred_all,y)
            self.loss_history.append(loss)

            if self.verbose == True and epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss = {loss:.4f}")

    def predict_proba(self,X):
        return self.sigmoid(np.dot(X,self.w) + self.b)

    def predict(self,X,threshold = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self):
        return self.w.copy(), self.b

#=======================
#       METRICS        =
#=======================

def confusion_matrix(y_true,y_pred):
    y_true  = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[TN,FP],[FN,TP]])

def accuracy(y_true,y_pred):
    return np.mean(y_pred == y_true)

def precision(y_true,y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP/(TP+FP+1e-13) #Tránh chia cho 0

def recall(y_true,y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP/(TP+FN+1e-13)

#====================================
# Train_test_split & standard scaler=
#====================================

def train_test_split_numpy(X, y, test_size=0.2, shuffle=True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def standard_scaler_fit_transform(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std==0] = 1
    return (X - mean) / std, mean, std

