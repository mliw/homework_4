import numpy as np
import seaborn as sns
WEIGHTS = np.array([0.6,0.4])


def generate_single_return_rate(mu_stocks = 0.065,mu_bonds = 0.017,sd_stocks = 0.195,sd_bonds = 0.075,rho = -0.15):
    mean = [mu_stocks,mu_bonds]
    vr_stocks = sd_stocks**2
    vr_bonds = sd_bonds**2   
    cov = rho*sd_stocks*sd_bonds
    cov_mat = [[vr_stocks,cov],[cov,vr_bonds]]
    return np.random.multivariate_normal(mean,cov_mat)
    

def asset_at_T(T,correlation):
    asset = 10000
    stock_bond = WEIGHTS*asset
    for i in range(T):
        return_rate = generate_single_return_rate(rho=correlation)
        stock_bond = stock_bond*(return_rate+1) #Allocation of 60/40
        stock_bond = sum(stock_bond)*WEIGHTS
    return sum(stock_bond)
    

def gen_sample(T,correlation,num):
    result = [asset_at_T(T,correlation) for i in range(num)]
    return np.array(result)
    

def draw_pic_and_cal(correlation,name):
    sample_1 = gen_sample(40,-correlation,2000)
    sample_2 = gen_sample(40,correlation,2000)
    mean_1 = np.round(np.mean(sample_1),2)
    mean_2 = np.round(np.mean(sample_2),2)
    std_1 = np.round(np.std(sample_1),2)
    std_2 = np.round(np.std(sample_2),2)
    
    import matplotlib.pyplot as plt
    figsize = 8, 9
    plt.subplots(figsize=figsize)  
    sns.set()
    sns.distplot(sample_1, kde=True,color="red")
    sns.distplot(sample_2, kde=True,color="blue") 
    plt.legend(["rho is "+str(-correlation)+" mean is {}".format(mean_1)+" std is {}".format(std_1),"rho is "+str(correlation)+" mean is {}".format(mean_2)+" std is {}".format(std_2)])
    plt.savefig("pic/"+name,dpi=400)
    plt.close()
    
    sns.set()
    sns.distplot(sample_1, kde=True,color="red") 
    plt.legend(["rho is "+str(-correlation)+" mean is {}".format(mean_1)+" std is {}".format(std_1)])
    plt.savefig("pic/negative_correlation_"+name,dpi=400)
    plt.close()
    
    sns.set()
    sns.distplot(sample_2, kde=True,color="blue") 
    plt.legend(["rho is "+str(correlation)+" mean is {}".format(mean_2)+" std is {}".format(std_2)])
    plt.savefig("pic/positive_correlation_"+name,dpi=400)
    plt.close()
    
if __name__=="__main__":
    
    for i in range(5):
        name = str(i)+".jpg"
        draw_pic_and_cal(0.3,name)
    

   