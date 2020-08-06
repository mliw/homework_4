import pandas as pd
import numpy as np
from glob import glob
import pickle
from tools import data_help
import warnings
warnings.filterwarnings("ignore")


if __name__=="__main__":
    
    # 1 Load trained model
    names = glob("lightgbm/*")
    with open(names[1],"rb") as f:
        model = pickle.load(f)    
    features = model["name"]
    train_x = data_help.train_x
    train_x = train_x.loc[:,features]
    train_y = data_help.train_y   
    """
    The following is train_x.mean()
    size            234637.743476
    empl_gr              3.206720
    leasing_rate        82.606371
    stories             13.584495
    age                 47.243983
    renovated            0.379529
    class_a              0.399924
    class_b              0.459463
    LEED                 0.006841
    green_rating         0.086775
    amenities            0.526602
    cd_total_07       1229.354193
    Gas_Costs            0.011336
    cluster_rent        27.497285
    dtype: float64
    """
    
    
    # 2 Define parameters
    for Gas_Costs in [2.891412e-02,0.011336]:
        print("="*60)
        print("Gas_Costs is {}".format(Gas_Costs))
        size = 250000
        empl_gr = 3.206720
        leasing_rate = 85
        stories = 15
        age = 0
        renovated = 0
        class_a = 0
        class_b = 1
        amenities = 1
        cd_total_07 = 200  
        cluster_rent = 28
        basic_para = {"size":size,"empl_gr":empl_gr,"leasing_rate":leasing_rate,"stories":stories,"age":age,"renovated":renovated,"class_a":class_a, \
                      "class_b":class_b,"amenities":amenities,"cd_total_07":cd_total_07,"Gas_Costs":Gas_Costs,"cluster_rent":cluster_rent}
        
        md = model["model"]
        md.fit(train_x,train_y)
        for leed,green in[[0,0],[0,1]]:
            tem_para = basic_para.copy()
            tem_para.update({"LEED":leed})
            tem_para.update({"green_rating":green})
            
            test_x = pd.DataFrame([],columns = train_x.columns,index = [0])
            for key in tem_para.keys():
                test_x.loc[0,key] = tem_para[key]
            print("green rate is {}".format(green))
            print("Corresponding house rent is {}".format(md.predict(test_x)[0]))
        print("="*60)      
        

      
    

   
