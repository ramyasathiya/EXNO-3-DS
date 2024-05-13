## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
#### STEP 1:Read the given Data.
#### STEP 2:Clean the Data Set using Data Cleaning Process.
#### STEP 3:Apply Feature Encoding for the feature in the data set.
#### STEP 4:Apply Feature Transformation for the feature in the data set.
#### STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
#### Name : KALPANA S
#### Ref No : 212222040069
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/fc676205-cd41-4ff1-810c-05ed1174f896)


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/ba618cc9-5b48-41af-9889-29bc41cbd6e2)



```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/417caa73-1322-4a56-af81-736a4ec8c189)


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/04885d62-50c1-480f-ae3c-08568172374b)


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/cfb922c7-8494-45b8-9519-f54999ddf17c)


```
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/aebf1256-0f49-4149-a76e-b9ebc5abaf4e)


```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/ef88024e-a579-4714-a779-c2dbd75b0c78)


```
pip install --upgrade category_encoders
```
![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/88f2d99c-315a-4508-bf7a-60c5374a5f43)


```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/cd6feeee-00ce-45a9-8c28-eba07e86ae1f)



```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/f10f049c-304e-448e-a96a-152250170dac)



```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/91e5344c-cf6c-49d4-96a1-e73237284876)



```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/5ed64258-82ff-49c3-af58-3c85e080cdbb)



```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/12ffaa4e-0b91-41f0-8181-a390f9fce9da)


```
df.skew()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/0e96b64f-1f2e-471c-b21a-31e26a7ddb60)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/25cdc061-2dbe-47c8-b862-954b0590efa6)


```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/d7c0016f-f32b-45c0-b333-85d77f32f72b)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/a89535c9-9890-4a3e-bffe-407f6f4d4c78)


```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/5d07851a-e341-4ef2-a986-68a7041463ab)




```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/a2f5ed85-7912-4fd5-a838-f0b3201a4819)



```
df.skew()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/3f0bb160-0e71-437a-bf84-79ffb0587dfd)



```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/1f99f9df-9f41-4f38-bde7-b2a958feea93)



```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/a60963a1-00a1-4de1-ac82-a6d10ce457a7)



```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/e8fa66cb-ff78-496c-bb69-0d904fe9a69e)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/f90300da-5c31-4136-a0a2-03b7c5af95c5)



```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/9249ebcf-fe7b-4699-adde-8bf7e89def8e)



```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/90325656-017f-4082-8eb7-ed129a86a289)



# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
