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

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-12)
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

#==================================
#       Oversampling (SMOTE)      =
#==================================
def smote_numpy(X,y,minority_class = 1 , k = 5, amount = 1.0):
    X_min = X[y == minority_class]
    n_min = len(X_min)
    if n_min == 0:
        return X,y
    n_new = int(n_min*amount)
    if n_new == 0:
        return X,y
    dist = np.full((n_min,n_min),np.inf)
    for i in range(n_min):
        diff = X_min - X_min[i]
        dist[i] = np.sqrt(np.sum(diff**2,axis = 1))
        dist[i,i] = np.inf

    k = min(k,n_min-1)
    neigh = np.argsort(dist,axis = 1)[:,:k]
    synthetic = []

    for _ in range(n_new):
        i = np.random.randint(0,n_min)
        j = np.random.choice(neigh[i])
        lam = np.random.rand()
        synthetic.append(X_min[i]+ lam*(X_min[j]-X_min[i]))

    X_syn = np.array(synthetic)
    y_syn = np.full(n_new,minority_class)

    return np.vstack([X,X_syn]),np.concatenate([y,y_syn])

#==========================
#      TRAIN/EVALUATE     =
#==========================
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_pred_proba": y_proba,
        "confusion_matrix": cm,
        "accuracy": accuracy(y_test, y_pred),
        "precision": precision(y_test, y_pred),
        "recall": recall(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "loss_history": model.loss_history,
    }
#===========================
#    FEATURE IMPORTANCE    =
#===========================

def feature_importance_from_weights(weights_tuple, feature_names):
    w, b = weights_tuple
    abs_w = np.abs(w)

    sorted_ids = np.argsort(-abs_w)

    results = []
    for idx in sorted_ids:
        results.append((feature_names[idx], w[idx], abs_w[idx]))

    return results


# =======================================
#         PRECISION-RECALL CURVE        =
# =======================================

def precision_recall_curve_manual(y_true, y_proba):

    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]
    y_proba_sorted = y_proba[order]

    tps = 0
    fps = 0
    fn = np.sum(y_true_sorted == 1)

    precisions, recalls, thresholds = [], [], []
    last_p = None

    for i in range(len(y_proba_sorted)):
        p_i = y_proba_sorted[i]

        if last_p is not None and p_i != last_p:
            precision_val = tps / (tps + fps + 1e-12)
            recall_val = tps / (tps + fn + 1e-12)
            precisions.append(precision_val)
            recalls.append(recall_val)
            thresholds.append(last_p)

        if y_true_sorted[i] == 1:
            tps += 1
            fn -= 1
        else:
            fps += 1

        last_p = p_i

    precision_val = tps / (tps + fps + 1e-12)
    recall_val = tps / (tps + fn + 1e-12)

    precisions.append(precision_val)
    recalls.append(recall_val)
    thresholds.append(last_p)

    return np.array(thresholds), np.array(precisions), np.array(recalls)


def auc_average_precision(recalls, precisions):
    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]

    auc = 0.0
    for i in range(1, len(r)):
        delta = r[i] - r[i - 1]
        auc += p[i] * delta
    return auc

def k_fold_split(n_samples, k):
    indices = np.arange(n_samples)
    folds = []
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    start = 0
    for fs in fold_sizes:
        end = start + fs
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
        start = end
    return folds

#==============================
#       CROSS-VAL-SCORE       =
#==============================
def cross_val_score_numpy(model_class, X, y, k=5, **model_params):
    accs, precs, recs, f1s = [], [], [], []
    folds = k_fold_split(len(y), k)

    for train_idx, val_idx in folds:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        accs.append(accuracy(y_val, y_pred))
        precs.append(precision(y_val, y_pred))
        recs.append(recall(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))

    return {
        "mean_acc": float(np.mean(accs)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall": float(np.mean(recs)),
        "mean_f1": float(np.mean(f1s)),
        "acc_list": accs,
        "precision_list": precs,
        "recall_list": recs,
        "f1_list": f1s,
    }

