import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def extract(df):
    mean = df["mean"][0]
    obs_ci_lower = df["obs_ci_lower"][0]
    obs_ci_upper = df["obs_ci_upper"][0]
    print("Mean value is {}, Upper bound is {}, Lower bound is {}".format(mean,obs_ci_upper,obs_ci_lower))
  

def ols_pre(re,value):
    print("="*60)
    print(value)
    arr = [[1,value]]
    predictions = re.get_prediction(arr)
    extract(predictions.summary_frame(alpha=0.05))    
    print("="*60)
    
    
if __name__=="__main__":
    
    # 1 Drawing pictures
    f = open("data/creatinine.csv")
    creatinine_data = pd.read_csv(f)
    sns.set()
    sns.regplot(x="age",y="creatclear",data=creatinine_data,label=True)
    plt.savefig("pic/regression_plot.jpg",dpi=600)
    plt.close()
    
    
    # 2 Get Prediction
    import statsmodels.api as sm
    x = np.array(creatinine_data.age)
    X = sm.add_constant(x)  
    y = creatinine_data.creatclear
    re = sm.OLS(y, X).fit()
    print(re.summary())
    ols_pre(re,55)
    ols_pre(re,40)
    ols_pre(re,60)