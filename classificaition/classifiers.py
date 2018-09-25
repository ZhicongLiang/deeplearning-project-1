'''
'''
import sys
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')

def load_data(feature_train, feature_test):
    '''
    loading the specific training and testing features
    '''
    y_test = pickle.load(open('../features/test_target.p','rb'))
    y_train = pickle.load(open('../features/train_target.p','rb'))
    if 'npy' in feature_train:
        x_train = np.load('../features/'+feature_train)
        x_test = np.load('../features/'+feature_test)
    elif '.p' in feature_train:
        x_train = pickle.load(open('../features/'+feature_train,'rb'))
        x_test = pickle.load(open('../features/'+feature_test,'rb'))
    return (x_train, y_train.reshape(y_train.size)), (x_test, y_test.reshape(y_test.size))

def accuracy(pred, target):
    return np.mean(pred==target)

def plot2d(x ,x_project, y, title, save=False):
    f = plt.figure(figsize=(8,6))
    plt.scatter(x_project[:,0], x_project[:,1], c=y, s=2)
    plt.colorbar()
    plt.title(title)
    plt.show()
    if save is True:
        f.savefig(title+'.jpg')

def LDA(n_component=2):
    '''
    Linear Discrimination Analysis
    component-- dimension after dimension reduction    
    '''
    clf = LinearDiscriminantAnalysis(n_components=n_component)
    clf.fit(x_train, y_train)
    x_test_project = clf.transform(x_test)
    plot2d(x_test, x_test_project, y_test, 'LDA\n%s'%feature_test.split('.')[0])
    y_test_pred = clf.predict(x_test)
    acc = accuracy(y_test_pred, y_test)
    return acc

def LogReg(C=1):
    '''
    Logistic Regression Classification
    C-- Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.
    '''
    clf = LogisticRegression(C=C, verbose=True)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    acc = accuracy(y_test_pred, y_test)
    return acc
    
def SVM():
    '''
    Support Vector Machine
    '''
    clf = svm.SVC(decision_function_shape='ovr',
                  verbose=True)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    acc = accuracy(y_test_pred, y_test)
    return acc

def RandomForest(max_depth=2, random_state=0):
    '''
    Random Forest
    '''
    clf = RandomForestClassifier(max_depth=max_depth, \
                                 random_state=random_state, verbose=True)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    acc = accuracy(y_test_pred, y_test)
    return acc

if __name__ == '__main__':
    
    feature_train = 'scatnet_train_feature_4_6.npy'
    feature_test = 'scatnet_test_feature_4_6.npy'
    (x_train, y_train), (x_test, y_test) = load_data(feature_train, feature_test)
    
    # LDA
    n_component = 2
    acc_lda = LDA(n_component=n_component)
    print('acc_lda:', acc_lda)
    
    # logistic regression
    C = 1
    acc_logreg = LogReg(C)
    print('acc_logreg:', acc_logreg)
    
    # svm
    acc_svm = SVM()
    print('acc_svm:', acc_svm)
    
    # random forest
    max_depth = 10
    random_state = 0
    acc_rand_forest = RandomForest(max_depth, random_state)
    print(acc_rand_forest)
    
    
    
    
    
    