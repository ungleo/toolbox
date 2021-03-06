from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# KS- table, para cada punto de corte las metricas
def ks_table(y_test,y_pred,decimal=2):
    """Genera una tabla con el k-s ,  precision , recall y f1-score para los distintos puntos de corte
        Args:
            y_test (array): valor real de Y
            y_pred (array): prediccion del modelo
            decimal (int): cantidad de decimales para mostrar en la tabla
        Returns:
           df: devuelve una tabla con el k-s ,  precision , recall y f1-score para los distintos puntos de corte
    """
    df_ks = pd.DataFrame(y_test)
    df_ks.columns=['target'] # rename de la columna para luego poder llamarla
    df_ks['pred']= y_pred
    df_ks['target0'] = 1 - df_ks['target']
    df_ks['bucket'] = df_ks['pred'].apply(lambda x: int(x*10**decimal)/(10**decimal) )
    grouped = df_ks.groupby('bucket', as_index = False)

    kstable = pd.DataFrame()
    kstable['prob_group'] =grouped.min()['bucket']
    kstable['min_prob'] = grouped.min()['pred']
    kstable['max_prob'] = grouped.max()['pred']
    kstable['events']   = grouped.sum()['target']
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / df_ks['target'].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / df_ks['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / df_ks['target'].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / df_ks['target0'].sum()).cumsum()
    
    kstable['TP'] = kstable.events.cumsum()
    kstable['FP'] = kstable.nonevents.cumsum()
    kstable['TN'] =  df_ks['target0'].sum() - kstable.nonevents.cumsum()
    kstable['FN'] =  df_ks['target'].sum() - kstable.events.cumsum()
    
    # precision = TP/(TP + FP)
    kstable['Precision_(1)'] = round(kstable['TP'] / (kstable['TP'] + kstable['FP'] ),2)
    # recall = TP / (TP + FN)
    kstable['recall_(1)'] = round(kstable['TP'] / (kstable['TP'] + kstable['FN'] ),2)
    
    # F1-score
    kstable['F1-score(1)'] = round ( 2* kstable['Precision_(1)'] *kstable['recall_(1)'] /( kstable['Precision_(1)']  + kstable['recall_(1)'] ) ,2)
    
    kstable['Precision_(0)'] = round(kstable['TN'] / (kstable['TN'] + kstable['FN'] ),2)
    kstable['recall_(0)'] = round(kstable['TN'] / (kstable['TN'] + kstable['FP'] ),2)
    kstable['F1-score(0)'] = round ( 2* kstable['Precision_(0)'] *kstable['recall_(0)'] /( kstable['Precision_(0)']  + kstable['recall_(0)'] ) ,2)
    
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100
    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    return kstable
    
# plot del ks



def pc_curve(labels,preds):
    precision, recall, thresholds = sklm.precision_recall_curve(labels, preds)
    plt.plot(recall, precision, marker='.', label='model')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

# mejor cut off
def best_cutoff(y_test, y_pred_prob, model_id):
    print('Best cutoff..')
    # comparo los y_test (true) con los y_pred_prob (scores)
    data = pd.DataFrame({'target': np.array(y_test), 'prediction': np.array(y_pred_prob)})
    data.sort_values(by=['prediction'], inplace=True, ascending=False)
    # counting 0s and 1s for data
    ones = sum(data['target'])
    zeros = len(data) - ones
    # we'll create columns in Data to get the churn and no_churn probs with the above defined function and then calculate the KS as the diff of probs
    data.reset_index(drop=True, inplace=True)
    data['target_reverse'] = data.target.apply(lambda x: 0 if x==1 else 1)
    data['cumsum_1'] = data.target.cumsum()
    data['cumsum_0'] = data.target_reverse.cumsum()
    data['probs_1'] = data['cumsum_1']/ones 
    data['probs_0'] = data['cumsum_0']/zeros 
    data['KS'] = data['probs_1'] - data['probs_0']
    data['quantile'] = data.index/len(data)
    
    # getting the probability where we maximize the KS metric
    max_ks = max(data['KS'])
    cutoff = float(max(data.loc[data.KS == max_ks,'prediction']))
    return max_ks, cutoff

def obtain_metrics(y_pred_prob, y_test, cutoff, max_ks):
    """
    Genera las metricas para un punto de corte puntual
    """
    y_pred = y_pred_prob >= cutoff
    accuracy = sklm.accuracy_score(y_test, y_pred)

    metrics = {
        'Cutoff': round(cutoff,3),
        'KS': round(max_ks,3),    
        'Accuracy': round(accuracy*100,2),
        'Precision (1)': round(sklm.precision_score(y_test, y_pred)*100,2),
        'Recall (1)': round(sklm.recall_score(y_test, y_pred)*100,2),
        'Mean absolute error': round(sklm.mean_absolute_error(y_test, y_pred),2),
        'mean squared error': round(sklm.mean_squared_error(y_test, y_pred),2),
        'Root mean squared error': round(np.sqrt(sklm.mean_squared_error(y_test, y_pred)),2)
    }
    cm, annot = cm_analysis(y_test, y_pred, [1,0], ymap=None)
    report = sklm.classification_report(y_test, y_pred)   
    return metrics

def cm_analysis(y_true, y_pred, labels=[1,0], ymap=None):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    tn, fp, fn, tp = sklm.confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[tp, fp], [fn ,tn]])
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d' % (p, c)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    return cm, annot


def print_confussion_matrix_and_report(cm, annot, report):
    print('> Confussion matrix')
    cm.index.name = 'Predicted condition'
    cm.columns.name = 'True condition'
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.show()   
    print('> Reporte:')
    print(report)

def plot_roc_curve(y_test, y_pred_prob, model_id):
    print('ROC curve & AUC..')
    fpr, tpr, thresholds = sklm.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve (AUC = %0.2f)' % sklm.roc_auc_score(y_test, y_pred_prob))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()
    #plt.savefig(f'{model_id} (ROC curve & AUC).png')

def get_importance_df(df_x,y,modelo): 
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(df_x.columns)
    imp_df["importance_gain"] = modelo.feature_importance(importance_type='gain')
    imp_df["importance_split"] = modelo.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, modelo.predict(df_x))
    return imp_df

# tabla con cuantiles por score
def quantile_table(y_test,y_pred,n_quantile=10):
    """Genera una tabla con la distribucion de target por cuantiles
        Args:
            y_test (array): valor real de Y
            y_pred (array): prediccion del modelo
            decimal (int): cantidad de decimales para mostrar en la tabla
        Returns:
           df
    """
    df_pred_y = pd.DataFrame(y_test)
    df_pred_y.columns=['target'] # rename de la columna para luego poder llamarla
    df_pred_y['pred']= y_pred
    df_pred_y['target0'] = 1 - df_pred_y['target']
    df_pred_y['bucket'] = pd.qcut(df_pred_y['pred'],n_quantile, labels = False)
    grouped = df_pred_y.groupby('bucket', as_index = False)

    table_group = pd.DataFrame()
    table_group['prob_group'] =grouped.min()['bucket']
    table_group['min_prob'] = grouped.min()['pred']
    table_group['max_prob'] = grouped.max()['pred']
    table_group['avg_prob'] = grouped.mean()['pred']
    table_group['events']   = grouped.sum()['target']
    table_group['nonevents'] = grouped.sum()['target0']
    table_group['obs'] = grouped.sum()['target'] + grouped.sum()['target0']

    table_group['prop_target'] = table_group['events'] / table_group['obs'] 
    table_group = table_group.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    table_group['event_rate'] = (table_group.events / df_pred_y['target'].sum()).apply('{0:.2%}'.format)
    table_group['nonevent_rate'] = (table_group.nonevents / df_pred_y['target0'].sum()).apply('{0:.2%}'.format)
    table_group['cum_eventrate']=(table_group.events / df_pred_y['target'].sum()).cumsum()
    table_group['cum_noneventrate']=(table_group.nonevents / df_pred_y['target0'].sum()).cumsum()
    table_group['KS'] = np.round(table_group['cum_eventrate']-table_group['cum_noneventrate'], 3) * 100
    return table_group
