import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from tools.feature_selection_ga import FeatureSelectionGA
from tools import data_help
import hyperopt
import pickle
N_FOLDS = 5


# N_FOLDS determines the outcome of this function
def produce_holdout(model,total_data_cache,individual):
    
    train_x = total_data_cache[0].copy()
    train_y = total_data_cache[1].copy()
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)
    individual = np.array(individual).astype(bool)
    train_x = train_x.loc[:,individual]
    
    kf = KFold(N_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_x,train_y)   
    result = []
    
    for train_index, test_index in kf.split(train_x, train_y):
        tem_train_x, tem_train_y = train_x.iloc[train_index,:], train_y.iloc[train_index,:]
        tem_test_x, tem_test_y = train_x.iloc[test_index,:], train_y.iloc[test_index,:] 
        tem_train_y = tem_train_y.values.reshape(-1)           
        model.fit(tem_train_x,tem_train_y)
        prediction = model.predict(tem_test_x)
        result.append(np.sqrt(mean_squared_error(tem_test_y,prediction)))
    
    return result


def nm_penalty(model,total_data_cache,individual):
    individual[10] = 1 # Incorporate Green-rating
    individual = np.array(individual).astype(bool)
    holdout = produce_holdout(model,total_data_cache,individual)
    result = np.mean(holdout)+np.std(holdout)
    return -result


def change_best_lgbm(best_p):
    best = best_p.copy()
    best["max_depth"]+=1  
    best["n_estimators"]+=1  
    best["n_jobs"]=1
    best["random_state"]=42
    
    return best


if __name__=="__main__":
    
    # 1 Load total_data_cache
    train_x = data_help.train_x
    train_y = data_help.train_y
    total_data_cache = [train_x,train_y]
    feature_names = train_x.columns
    

    # 2 Define model and parameters
    robust_lightgbm = make_pipeline(RobustScaler(), LGBMRegressor(random_state=42,objective="huber",n_jobs=2))
    LGBMRegressor_dic = {
        'learning_rate': hyperopt.hp.uniform('learning_rate',0,2),
        'reg_alpha': hyperopt.hp.uniform('reg_alpha',0,2),
        'reg_lambda': hyperopt.hp.uniform('reg_lambda',0,2),
        'n_jobs': hyperopt.hp.choice('n_jobs',[1]),
        'random_state': hyperopt.hp.choice('random_state',[42]),
        'max_depth': hyperopt.hp.randint('max_depth',11)+1, 
        'n_estimators': hyperopt.hp.randint('n_estimators',250)+1,    
        'subsample': hyperopt.hp.uniform('subsample',0,1)
    }

    
    # 3 Genetic Algorithm for feature selection
    key = "lightgbm"
    probability = 0.7
    bench_model = robust_lightgbm
    length_of_features = train_x.shape[1]
    
    
    # 4 Start evolving and tunning
    populations = 400 
    generations = 7    
    selector = FeatureSelectionGA(bench_model,total_data_cache,length_of_features,nm_penalty,probability)
    print("35 generations are required to find the best individual. Please wait~~")
    selector.generate(populations,ngen=generations,cxpb=0.1,mutxpb=0.8)

    record = selector.best_generations
    record_score = float("inf")
    result_collect = []
    count = 0
    for i in range(len(record)):
        final_features = record[i]
        final_score = -record[i].fitness.values[0]
        if final_score < record_score:
            record_score = final_score   
            final_features = np.array(final_features).astype(bool)
            final_feature_name = feature_names[final_features]
            
            result = {}
            result.update({"name":final_feature_name})
            result.update({"model":bench_model})
            result.update({"score":record_score}) 
            saved_str = str(count)+"_cv_score_"+str(record_score)
            result.update({"saved_str":saved_str})
            result_collect.append(result)
            count+=1            
            
    # 4.1 Save original models and tunned models
    items = result_collect[-1]
    saved_str = items["saved_str"]        
    with open(key+"/"+saved_str+".pkl","wb") as f:
        pickle.dump(items,f)

    # Tunning model_para
    tunning_train_x = train_x.loc[:,items["name"]] 
    def objective(param):
        tuning_pipeline = make_pipeline(RobustScaler(),LGBMRegressor(**param))
        loss = -nm_penalty(tuning_pipeline,[tunning_train_x,train_y],np.ones(tunning_train_x.shape[1]))
        return loss  
    trials = hyperopt.Trials()
    best = hyperopt.fmin(objective,
        space=LGBMRegressor_dic,
        algo=hyperopt.tpe.suggest,
        max_evals=2000,
        trials=trials)      
    
    best_para = change_best_lgbm(best)
    tunned_result = {}
    tunned_result["name"] = items["name"]
    tunned_result["model"] = make_pipeline(RobustScaler(),LGBMRegressor(**best_para))
    tunned_result["score"] = -nm_penalty(tunned_result["model"],[tunning_train_x,train_y],np.ones(tunning_train_x.shape[1]))
    tunned_result["saved_str"] = saved_str+"_"+str(tunned_result["score"])
    
    with open(key+"/"+tunned_result["saved_str"]+".pkl","wb") as f:
        pickle.dump(tunned_result,f)        
    

   
