# pip install pandas
# pip install

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

dfs = [df_test, df_train]

for df in dfs:
    def make_bdate(data):
        data = str(data).split(".")
        if len(data) > 2:
            return int(data[2])
        return -1

    df["bdate"] = df["bdate"].apply(make_bdate)

    def make_education_form(data):
        if data == "Full-time":
            return 1
        elif data == "Part-time":
            return 2
        elif data == "Distance Learning":
            return 3
        else:
            return -1

    df["education_form"] = df["education_form"].apply(make_education_form)

    df.drop(["graduation", "has_photo", "education_status","langs","life_main","people_main","city","last_seen","occupation_type","occupation_name","career_start","career_end"], axis=1, inplace=True)




df_train.info()



X_train= df_train.drop(["id","result"], axis=1)
y_train = df_train["result"]

model= KNeighborsClassifier()
model.fit(X_train,y_train)

X_test = df_test.drop(["id"], axis=1)
y_test = model.predict(X_test)

print(y_test)

df_result = pd.DataFrame({"ID":df_test["id"], "result": y_test })

df_result.to_csv("result.csv", index =False)