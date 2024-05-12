# EXP-04-Feature Scaling and Selection
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![326256682-6e19decc-aca1-4c5d-a5a8-808e5fa87e4b](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/695ab356-a53f-489f-a8e7-b571fbf4ac22)

```
data.isnull().sum()
```
![326256707-271aa063-6a80-4c28-811a-7171a56ff2b9](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/98b68998-cc53-46c4-af34-4a59f4252fd8)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![326256739-b47d1c32-8a17-45e2-a2e8-bae88ba106e9](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/2aafd6b9-1781-4fdc-9b9b-01242d97fe1f)

```
data2=data.dropna(axis=0)
data2
```
![326256755-45dae742-c63c-4a66-9f22-528abdeb80c7](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/e22657bf-ed52-4263-a24c-659aeffb8f1a)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![326256774-793856c0-3130-491d-913c-079c91df14ec](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/1c8cb30f-47ab-41f6-875c-8b25d99415c7)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![326256795-db98fa80-01d2-48b9-9602-249ed3de56e5](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/4e75d8a5-ea3a-4f1a-b9c3-95855db00ef6)

```
data2
```
![326256819-5a0bcd74-1b11-43fe-8a06-5ebdec810fb3](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/1f2411d8-6469-4f60-8280-bbfacb0eb5b2)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
 ```
![326256843-15759418-8554-4894-b17d-189bf8738a23](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/699c0c94-3bae-4eea-bbd9-e0f58c80bc87)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![326256885-e6f4ca22-5ff3-4931-928a-41f32f27ff05](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/a7c91966-0a6f-4731-87cf-cae430494425)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![326256922-b76be506-0fe7-45d9-a7b7-c0ab9d85dc93](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/ce99e0a7-e3db-43df-a284-545b4f675809)

```
y=new_data['SalStat'].values
print(y)
```

![326256947-d347c35b-17f2-464b-937a-6f250f1eb9b6](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/44cf00b9-0817-47e8-bcfe-d2e4f699465f)

```
x=new_data[features].values
print(x)
```

![326256970-a6c71ac3-9d02-489e-b8b9-f608a8dcd4a9](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/e5b0a710-5876-4cbd-80b6-5e36fbdecc86)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![326256990-be5771c2-c53a-4abe-8093-b8a133ab3a28](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/0f24a240-db80-4c89-9e6f-370c4a2b88aa)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![326257036-5bc37c28-adcf-42bb-bba3-af3968706a0d](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/c4687cca-938a-4f19-be97-4ff3ac5d6ba5)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![326257060-12854589-eb9c-498d-935f-a787d3dcfd1b](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/1780df45-6e16-402e-9009-0f78bb033519)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![326257088-73ca3858-916c-4764-a1c0-df6266f07f48](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/b841c9e4-7114-4b07-b779-47c86ff4aa58)

```
data.shape
```

![326257125-3e7ec22d-61c7-46e9-81b5-b5de3c0d8e54](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/90a0e645-8e55-40fa-99bc-0d0a2039a391)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![326257148-588e94d8-2242-4f82-917c-767918b6d72c](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/72e4293e-7e07-4515-8b49-86f9820532c8)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![326257169-d9652b0e-b598-4474-b929-2e21eae2748d](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/098195c1-abc1-4e84-addb-ec2cbdf86b0e)

```
tips.time.unique()
```
![326257198-85335a63-e7e1-4769-a559-292172b20e10](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/d2e1725e-5135-4e21-81b8-5af4a297e23d)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![326257240-34568951-2d9e-4f97-861a-3e610b496809](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/5f16af27-85cd-4737-8652-5e7528103ec0)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![326257259-a4521743-8b08-4407-aa79-2f71e04ff4b7](https://github.com/Swetha733N/EXNO-4-DS/assets/122199934/c2a13f7e-5661-4944-be90-0474b220ff4d)



# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
