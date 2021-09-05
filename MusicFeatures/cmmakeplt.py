from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import librosa
import librosa.display
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')

import os
general_path = './data'


data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:] 
data.head()

y = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label'] #select all columns but not the labels

#### NORMALIZE X ####

# Normalize so everything is on the same scale. 

cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

# new data frame with the new scaled data. 
X = pd.DataFrame(np_scaled, columns = cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def model_assess(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')
    
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report,confusion_matrix
import itertools  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, name='Gaussnb'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.figure(figsize = (9, 7), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm)
#     ax.set_ylim(len(cm)-0.5, -0.5)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)  # rotation=45
    plt.yticks(tick_marks, classes)
    plt.ylim(len(cm) - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists(f'./resulImgs/{name}cm.jpg'):
        plt.savefig(f'./resulImgs/Cm{name}.jpg')

def confusion_matrix_(y_pred, y):
    cm = confusion_matrix(y, y_pred, labels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    ss0 = "TP, TN, FP, FN: " + str(TP.mean()) + " " +  str(TN.mean()) + " " + str(FP.mean()) + " " + str(FN.mean()) + '\n'
    f.write(ss0)
    print("TP, TN, FP, FN: ", TP.mean(), TN.mean(), FP.mean(), FN.mean())
    TPR = TP/(TP+FN) 
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    REC = TP/(TP+FN)
    ACC = (TP+TN)/(TP+FN+TN+FP)
    ss = 'Sensivity: %.2f%%'%(TPR.mean()*100) + '\n' + \
         'Specificity: %.2f%%'%(TNR.mean()*100) + '\n' + \
         'Precision: %.2f%%'%(PPV.mean()*100) + '\n' + 'Pecall: %.2f%%'%(REC.mean()*100) + '\n'
         
    f.write(ss)
    
    print('Sensivity: %.2f%%'%(TPR.mean()*100))
    print('Specificity: %.2f%%'%(TNR.mean()*100))
    print('Precision: %.2f%%'%(PPV.mean()*100))
    print('Pecall: %.2f%%'%(REC.mean()*100))
#     print('Accuracy: %.2f%%'%(ACC.mean()*100))
#     print('F1:%.2f%%'%(100*(2*PPV.mean()*REC.mean())/(PPV.mean()+REC.mean())))
        
def CalculationResults(val_y,y_val_pred,simple = False,\
                       target_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], name='SGD'):
    # 计算检验
    F1_score = f1_score(val_y,y_val_pred, average='macro')
    if simple:
        return F1_score
    else:
        acc = accuracy_score(val_y,y_val_pred)
        recall_score_ = recall_score(val_y,y_val_pred, average='macro')
        confusion_matrix_ = confusion_matrix(val_y,y_val_pred)
        class_report = classification_report(val_y, y_val_pred)
        ss1 = 'f1:%.2f%%'%(F1_score*100) + '\n' + 'Accuarcy:%.2f%%'%(acc*100) + '\n'
        f.write(ss1)
        print('f1:%.2f%%'%(F1_score*100))
        print('Accuarcy:%.2f%%'%(acc*100))
        print('f1_score:',F1_score,'\nACC_score:',acc,'\nrecall:',recall_score_)
#         print('\n----class report ---:\n',class_report)
#         print('----confusion matrix ---:\n',confusion_matrix_)

        # 画混淆矩阵
            # 画混淆矩阵图
        plt.figure()
        plot_confusion_matrix(confusion_matrix_, classes=target_names,
                              title=f'Confusion matrix of {name}', name=name)
#         plt.show()
        return F1_score,acc,recall_score_,confusion_matrix_,class_report
    
f = open('result.txt', 'w')


# f.write('Naive_Bayes'+'\n')
# print("\nNaive_Bayes")
# nb = GaussianNB()
# model_assess(nb, "Naive Bayes")
# preds = nb.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='Naive Bayes')


# f.write('SGD'+'\n')
# print("\nSGD")
# sgd = SGDClassifier(max_iter=5000, random_state=0)
# model_assess(sgd, "Stochastic Gradient Descent")
# preds = sgd.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='SGD')

# f.write('KNN'+'\n')
# print("\nKNN")
# knn = KNeighborsClassifier(n_neighbors=19)
# model_assess(knn, "KNN")
# preds = knn.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='KNN')

# f.write('Decission trees'+'\n')
# print("\nDecission trees")
# tree = DecisionTreeClassifier()
# model_assess(tree, "Decission trees")
# preds = tree.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='Decission trees')


# f.write('Random Forest'+'\n')
# print("\nRandom Forest")
# rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
# model_assess(rforest, "Random Forest")
# preds = rforest.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='Random Forest')


# f.write('svm'+'\n')
# print("\nsvm")
# svm = SVC(decision_function_shape="ovo")
# model_assess(svm, "Support Vector Machine")
# preds = svm.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='SVM')


# f.write('Logistic Regression'+'\n')
# print("\nLogistic Regression")
# lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
# model_assess(lg, "Logistic Regression")
# preds = lg.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='Logistic Regression')


# f.write('Neural Nets'+'\n')
# print("\nNeural Nets")
# nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
# model_assess(nn, "Neural Nets")
# preds = nn.predict(X_test)
# confusion_matrix_(y_test, preds)
# CalculationResults(y_test, preds, name='Neural Nets')


f.write('Cross Gradient Booster'+'\n')
print("\nCross Gradient Booster")
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, "Cross Gradient Booster")
preds = xgb.predict(X_test)
confusion_matrix_(y_test, preds)
CalculationResults(y_test, preds, name='Cross Gradient Booster')



f.write('Cross Gradient Booster (Random Forest)'+'\n')
print("\nCross Gradient Booster (Random Forest)")
xgbrf = XGBRFClassifier(objective= 'multi:softmax')
model_assess(xgbrf, "Cross Gradient Booster (Random Forest)")
preds = xgbrf.predict(X_test)
confusion_matrix_(y_test, preds)
CalculationResults(y_test, preds, name='Cross Gradient Booster (Random Forest)')



f.close()