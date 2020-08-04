import pandas as pd


f = open("data/greenbuildings.csv")
tem_data = pd.read_csv(f)
tem_data.index = tem_data.CS_PropertyID
del(tem_data["CS_PropertyID"])

feature_names = list(tem_data.columns)
feature_names.remove("Rent")
feature_names.remove("cluster")
target_names = ["Rent"]

train_x = tem_data.loc[:,feature_names]
train_y = tem_data.loc[:,target_names]