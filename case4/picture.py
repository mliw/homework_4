import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
N_FOLDS = 5


def produce_holdout(degree,total_data_cache,rd):
    train_x = total_data_cache[0].copy()
    train_y = total_data_cache[1].copy()
    kf = KFold(N_FOLDS, shuffle=True, random_state=rd)
    kf.get_n_splits(train_x,train_y)   
    result = []
    for train_index, test_index in kf.split(train_x, train_y):
        tem_train_x, tem_train_y = train_x[train_index], train_y[train_index]
        tem_test_x, tem_test_y = train_x[test_index], train_y[test_index] 
        z1 = np.polyfit(tem_train_x, tem_train_y,degree) 
        p1 = np.poly1d(z1)
        prediction = p1(tem_test_x)
        result.append(np.sqrt(mean_squared_error(tem_test_y,prediction)))
    return result


def nm_penalty(degree,total_data_cache):
    result  = []
    for rd in range(500):
        holdout = produce_holdout(degree,total_data_cache,rd)
        penalty = np.mean(holdout)+np.std(holdout)
        result.append(penalty)
    return np.mean(result)


if __name__=="__main__":
    
    # 1 Drawing basic pictures
    f = open("data/milk.csv")
    milk = pd.read_csv(f)
    milk["log_price"] = np.log1p(milk["price"])
    sns.set()
    sns.regplot(x="sales",y="price",data=milk,label=True)
    plt.savefig("pic/price.jpg",dpi=600)
    plt.close()

    
    # 2 Training on data
    x = milk.sales
    y = milk.price 
    total_data_cache = [x,y]
    
    collection = []
    for degree in range(7):
        collection.append([degree,nm_penalty(degree,total_data_cache)])
    pic = pd.DataFrame(collection,columns=["degree","cross-validation-error"])    
    sns.set()
    sns.relplot(x="degree",y="cross-validation-error",data=pic,label=True,kind="line")
    plt.savefig("pic/degree.jpg",dpi=600)
    plt.close()
 
    
    # 3 Get coefficients
    z1 = np.polyfit(x, y,2) 
    p1 = np.poly1d(z1)
    print(p1)
   
    
    # 4 Draw figures
    based_x = np.linspace(0,100,1000)
    based_y = np.array([p1(items) for items in based_x])
    
    sns.set()
    sns.relplot(x="sales",y="price",data=milk,label=True)
    sns.lineplot(x=based_x,y=based_y,legend=False,color="red")
    plt.savefig("pic/add.jpg",dpi=600)
    plt.close()
    
    
    