from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

import dill as pickle
import datetime

def load_datasets(test_size_val=0.25,dev_size_val=0.2):
    dataset = load_files('review_polarity/txt_sentoken', shuffle=False)
    docs_traindev, docs_test, y_traindev, y_test = train_test_split(
        dataset.data, dataset.target, test_size=test_size_val, random_state=42)
    test = (docs_test, y_test)
    docs_train, docs_dev, y_train, y_dev = train_test_split(
        docs_traindev, y_traindev, test_size=dev_size_val, random_state=42)
    train = docs_train, y_train
    dev = docs_dev, y_dev
    return train, dev, test


from sklearn import metrics

    
def print_eval(model, X, y_true):
    y_pred = model.predict(X)
    acc = metrics.accuracy_score(y_true, y_pred)
    print('accuracy\t{:2.2f}\n'.format(acc))
    print(metrics.classification_report(y_true, y_pred, target_names=['neg', 'pos']))
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)


def eval(model, X, y_true):
    y_pred = model.predict(X)
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return {'acc': acc, 'f1': f1}


def print_short_eval(model, X, y_true):
    res = eval(model, X, y_true)
    print('accuracy\t{acc:2.2f}\tmacro f1\t{f1:2.2f}'.format(**res))

    
import pickle


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


import os
#from sklesarn.datasets import load_files
#from sklearn.model_selection import train_test_split

def load_datasets_unlabeled_test(test_size_val=0.10):
    dataset = load_files('review_polarity_competition/reviews_sentoken', shuffle=False)
    docs_train, docs_dev, y_train, y_dev = train_test_split(
        dataset.data, dataset.target, test_size=test_size_val, random_state=42)
    dirname = "review_polarity_competition/test_reviews_sentoken"
    test = []
    # I do this to keep the files in numeric order
    for fname in range(len(os.listdir(dirname))):
        fname = str(fname) + ".txt"
        with open(os.path.join(dirname, fname)) as fd:
            test.append(fd.read())
    train = docs_train, y_train
    dev = docs_dev, y_dev
    return train, dev, test

def save_results(fname, labels):
    with open(fname, 'w') as f:
        f.write("Id,Category\n")
        for i,l in enumerate(labels):
            f.write(str(i) + ".txt," + str(l) + "\n")

import pandas as pd            
def get_baseline():
    pd_out=pd.read_csv('review_polarity_competition/results_baseline.csv')
    return pd_out



def get_basline_versus_err(model,test):
    pd_out=get_baseline()
    y_test_baseline=pd_out['Category']
    id_baseline=pd_out['Id']
    y_test_pred=model.predict(test)
    pd_out['y_test_pred']=y_test_pred
    try:
        y_test_pred_proba=model.predict_proba(test)
        pd_out['y_test_pred_proba']=y_test_pred_proba
    except:
        pass
    errors = []
    for x, i ,y1, y2 in zip(test, id_baseline ,y_test_baseline, y_test_pred):
        if y1 != y2:
            errors.append({
                'item': x,
                'txt': i,
                'baseline': y1,
                'baseline_pred': y2})

    errdf = pd.DataFrame(errors)
    return errdf,pd_out

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
def print_plot_classification_report(model,X_case,Y_case,Case,plotFlag=False,classes_list=['NO_CASE','CASE']):
    print(Case)
    print("Reporte de clasificación: ", end="\n\n")
    y_true, y_pred = Y_case, model.predict(X_case)
    print(classification_report(y_true, y_pred), end="\n\n")
    print("================================================", end="\n\n")
    if plotFlag:
        plot_confusion_matrix(confusion_matrix(y_true, y_pred),
                      classes=classes_list, title="Matriz de confusión para "+ Case)
        plt.show()
        

        
def print_plot_classification_report_no_model(Y_case,Y_pred,Case,plotFlag=False,classes_list=['NO_CASE','CASE']):
    print(Case)
    print("Reporte de clasificación: ", end="\n\n")
    y_true, y_pred = Y_case, Y_pred
    print(classification_report(y_true, y_pred), end="\n\n")
    print("================================================", end="\n\n")
    if plotFlag:
        plot_confusion_matrix(confusion_matrix(y_true, y_pred),
                      classes=classes_list, title="Matriz de confusión para "+ Case)
        plt.show()        
        
        
        
        
        
def print_plot_classification_report_keras(model,X_case,Y_case,Case,plotFlag=False,classes_list=['NO_CASE','CASE']):
    print(Case)
    print("Reporte de clasificación: ", end="\n\n")
    
    Y_pred_prev=model.predict(X_case)
    
    y_true, y_pred = Y_case, np.argmax( Y_pred_prev ,axis=1)
    print(classification_report(y_true, y_pred), end="\n\n")
    print("================================================", end="\n\n")
    if plotFlag:
        plot_confusion_matrix(confusion_matrix(y_true, y_pred),
                      classes=classes_list, title="Matriz de confusión para "+ Case)
        plt.show()        
        
        
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(10,10)) #Agregado by me
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.figure(figsize=(10,10)) #Agregado by me
    plt.tight_layout()
    plt.ylabel('Etiqueta correcta')
    plt.xlabel('Etiqueta predicha')
    
def iterate_pipeline_params(pipeline,params_list,X_train,y_train,X_dev, y_dev):
    results = []
    for params in params_list:
        # TODO: add progress bar!
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        result = eval(pipeline, X_dev, y_dev)

        results.append({
            **result,
            **params,
        })
    return results



def print_GridSearch_params(model,Case,fullprint=False):
    print(Case)
    print("Mejor conjunto de parámetros para GridSearch en "+  Case +" :")
    print(model.best_params_, end="\n\n")
    if fullprint:
        print("Puntajes de la grilla:", end="\n\n")
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("Exactitud: %0.3f (+/-%0.03f) para los parámetros %r" % (mean, std ** 2, params))

def pd_csv_comp(file1,file2,cols=['Category','Category','Id','Id']):
    pd_1=pd.read_csv(file1)
    pd_2=pd.read_csv(file2)
    aux=[]
    for x,y,z,v in zip(pd_1[cols[0]],pd_2[cols[1]],pd_1[cols[2]],pd_2[cols[3]]):
        if x!=y:
            aux.append([x,y,z,v])
    return aux,pd.DataFrame(aux,columns=cols)


def save_to_pickle(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_keras_model_history_params(model,params_list_tuple):
    dOut={}
    dOut['history']=model.history.history
    dOut['model_params']=model.history.params
    for k,y in params_list_tuple:
        dOut[k]=y
    return dOut

def get_time_str():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')