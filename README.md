## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

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
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/765aef66-e448-4589-bdb4-713061923e9b)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/540fb927-b65b-421f-b387-72066567fa5a)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/df28587f-799f-443c-ba96-e46b6a50e7b9)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/ea6bee98-a286-4398-81bb-5d0a6529812f)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/902de21e-df37-4c5f-bebb-4f26fff3bc8c)

```
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/bb46fb33-9d91-4bce-8ab5-25eacd62abfc)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/8cceee9b-d4ce-4fc1-9da6-6f7a169e60ae)

```
pip install --upgrade category_encoders
```
![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/8e9aa278-d264-46b7-8135-4aaaef4c26c8)


```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/2a47bd12-d412-43d1-8dda-0dd8f85da92a)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/d20bcb03-feb0-45e2-9750-39f2cdc1b0ec)


```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/ec4ab0aa-31ae-4ee1-bb76-94979c247cd2)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/1dc0652b-50c1-4036-9998-f09478b5d62c)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/b04502e8-d026-4ee0-b9e7-7ad033364763)


```
df.skew()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/4a039b4d-54e5-4966-8fa9-cd68f2d70eee)


```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/c329a4c1-e6a6-41ce-a03b-b04383c913ff)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/1f266014-eca7-4c22-8dfb-eb1ce778d379)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/ed6d6bc1-2e8c-4a37-b5d2-5b3162950e7c)

```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/55a2c46c-4f00-47d8-8e30-b9189b84dca9)


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/356a427c-ab39-4893-ac06-d81969485f88)


```
df.skew()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/862a1998-0eae-457c-aefe-01832c6a992b)


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/c39c6ba0-f7ed-4565-bc7b-6e71932ce09c)


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/4fad9e73-8a18-44f7-8453-6b6ba7b65a5f)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/7ffb49c3-5df5-4872-bd54-672c3789dc44)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/c7d671c2-7546-41c5-8f54-33c0cdab9276)


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/10f4c41a-5741-452e-ab0b-d6c02b139f5f)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/Kalpanareshma/EXNO-3-DS/assets/122040453/6d118c01-e4a3-4ee6-b31e-8b688d00d10e)



# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
