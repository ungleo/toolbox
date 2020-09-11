import numpy as np
# limpiar data set
def clean_dataset(df, categorical_cols = None,canary=0, vars_to_drop=None):
    
    if vars_to_drop!=None:
        df.drop(vars_to_drop, axis=1, inplace=True)
    
    if categorical_cols != None:
        df[df.columns.difference(categorical_cols)] = df[df.columns.difference(categorical_cols)].astype('float')

        for c in categorical_cols:
            df[c] = df[c].astype('str')
            df[c] = df[c].fillna('undefined')

        for c in categorical_cols:
            label = LabelEncoder()
            label.fit(list(df[c].values))
            df.loc[:, c] = label.transform(list(df[c].values))
        df[df.columns.difference(categorical_cols)] = df[df.columns.difference(categorical_cols)].fillna(0)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols_num = df.select_dtypes(include=numerics).columns
    
    #df['device_name'] = df['device_name'].map(lambda x : clean_text(x))
    #Excluyo el target para que no aplique transf. log

    #df[cols_num] = df[cols_num].applymap(lambda x: np.log1p(x))
    if canary>0:
        print('canary features')
        for c in range(1,canary):
            df['canary_{}'.format(c)] = np.random.uniform(0,1,len(df))
    
    return df


# optimizacion bayesiana

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization
import lightgbm as lgb

def bayes_parameter_opt_lgb(train_x, train_y, valid_x, valid_y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=100, output_process=False):
    
    # prepare data
    train_data = lgb.Dataset(data=train_x, label=train_y, free_raw_data=False)
    valid_data =lgb.Dataset(data=valid_x, label=valid_y, free_raw_data=False)
    
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, bagging_freq, lambda_l1, lambda_l2, min_split_gain, min_child_weight, learning_rate):
        params = {'application':'binary', 'metric':'auc','verbose': -1}
        params["learning_rate"] = learning_rate 
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['bagging_freq'] = int(round(bagging_freq))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        
        m = lgb.train(params, train_data, valid_sets=[train_data,valid_data], num_boost_round=100,
                      early_stopping_rounds=50, verbose_eval=False)
        return m.best_score['valid_1']['auc']
    
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (10, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'bagging_freq': (3, 8),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'learning_rate': (0.001,0.1),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    i = np.argmax([val['target'] for val in lgbBO.res])
    
    # clean some of the params
    params = lgbBO.res[i].copy()
    
    params['params']['num_leaves'] = int(params['params']['num_leaves'])
    params['params']['max_depth'] = int(params['params']['max_depth'])
    params['params']['metric'] = 'auc'
    params['params']['application'] = 'binary'
    
    # return best parameters
    return params
