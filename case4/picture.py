import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def check_linear(df,x_name,y_name):
    print("="*79)
    print("x:"+x_name,"y:"+y_name)
    import statsmodels.api as sm
    x = np.array(df[x_name])
    X = sm.add_constant(x)  
    y = df[y_name]
    re = sm.OLS(y, X).fit()
    print(re.summary())
    
    
if __name__=="__main__":
    
    # 1 Drawing basic pictures
    f = open("data/milk.csv")
    milk = pd.read_csv(f)
    milk["log_price"] = np.log(milk["price"])
    milk["log_sales"] = np.log(milk["sales"])
    

    # 2 Print summary
    check_linear(milk,"price","sales")
    check_linear(milk,"log_price","sales")    
    check_linear(milk,"log_price","log_sales")   
    check_linear(milk,"price","log_sales")     
    
    
    # 3 Draw figures
    sns.set()
    sns.regplot(x="log_price",y="log_sales",data=milk,label=True)
    plt.savefig("pic/linear_log.jpg",dpi=600)
    plt.close()
    
    
    # 4 Draw figures regarding exponential function
    def func(x):
        return np.exp(4.7206)*x**(-1.6186)    
    based_x = np.linspace(1,5,1000)
    based_y = np.array([func(items) for items in based_x])
    
    sns.set()
    sns.relplot(x="price",y="sales",data=milk,label=True)
    sns.lineplot(x=based_x,y=based_y,legend=False,color="red")
    plt.savefig("pic/fitted.jpg",dpi=600)
    plt.close()
    
