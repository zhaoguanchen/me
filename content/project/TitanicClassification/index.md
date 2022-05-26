---
title: Titanic - Machine Learning from Disaster
subtitle: "This is a Kaggle competition named 'Titanic - Machine Learning from Disaster'. "
date: 2021-06-22T01:28:00.000Z
draft: false
featured: false
tags:
  - Machine Learning
categories:
  - Machine Learning

links:
  - icon: github
    icon_pack: fab
    name: Follow
    url: https://github.com/zhaoguanchen/Machine-Learning
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---

# 1. Introduction

## 1.1 Description

On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.  

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.  

Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.  

Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.

## 1.2 Data Population

According to the problem, the data population is every passenger in the Titanic shipwreck.  
Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).

# 2. Data

## 2.1 Kaggle

**Kaggle account:**
GuanchenZhao, DeyiZhou


---
**Comments**  
*Are you using a combined account? Please use the combined account to submit the work in stage-2*    


**<font color=red>Reply</font>**    
<font color=red>Two separate accounts are written in stage-1. Later, we will submit with ***GuanchenZhao***</font>.




## 2.2 Data Overview

There are three files in the data: (1) **train.csv**, (2) **test.csv**, and (3) **gender_submission.csv**.

#### 2.2.1 Train Data

train.csv contains the details of a subset of the passengers on board (891 passengers, to be exact -- where each passenger gets a different row in the table).    

For instance, the first passenger listed in train.csv is Braund, Mr. Owen Harris. He was 22 years old when he died on the Titanic.

import and show the train data.


```python
# data analysis and wrangling
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

train_df = pd.read_csv('data/train.csv')
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
print(train_df.columns.values)
```

    ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
     'Ticket' 'Fare' 'Cabin' 'Embarked']


The columns are as follows.


|  Variable | Definition | Value | 
|  ----  | ----  |----  |
|  PassengerId  | key  |number  |
| survival  | Survival |0 = No, 1 = Yes|
| pclass  | Ticket class |1st = Upper  ,  2nd = Middle,   3rd = Lower|
| sex  | sex |male, female |
| Age  | Age | Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5|
| sibsp  | family relations| 1= Sibling or Spouse, 0=else  |
| parch  |  family relations  |1= Parent or Child,0=else |
| ticket  | Ticket number | Ticket number |
| fare  | Passenger fare |number |
| cabin  |  Cabin number |Cabin number |
| embarked  | Port of Embarkation |C = Cherbourg, Q = Queenstown, S = Southampton |

#### 2.2.2 Test Data

The test.csv file is data that will be used to test the model. 

Import and show the train data.


```python
test_df = pd.read_csv('data/test.csv')

test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



The columns are as same as cloumns in train.csv. Note that test.csv does not have a "Survived" column - this information is hidden.

#### 2.2.3 Gender Submission

The gender_submission.csv file is provided as an example that shows the structure of predictions.  It predicts that all female passengers survived, and all male passengers died. 

the file contaions two columns.
*   "PassengerId": the IDs of each passenger from test.csv.
*   "Survived": the result.

Import and show the train data.


```python
gender_submission_df = pd.read_csv('data/gender_submission.csv')

gender_submission_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 2.3 Data Wrangling

Check missing values


```python
missingvalues_train = train_df.isnull().sum()
missingvalues_train[missingvalues_train>0]
```




    Age         177
    Cabin       687
    Embarked      2
    dtype: int64



### 2.3.1 Structure

**Data Type**

We know that there are several types of data structures and we will distinguish the data in the table into the structure for subsequent data analysis.

Data Type :
* Quantitative Data - Age, Sibsp, Parch, Fare  
* Categorical Data (Nominal) - PassengerId, Ticket, Cabin, Embarked, Survived, Sex, Name
* Categorical Data (Ordinal) - Pclass

**How big is the data?**


```python
print( "train.csv file size is", len(train_df))
```

    train.csv file size is 891



```python
print( "train.csv file size is", len(train_df))
```

    train.csv file size is 891



```python
print( "gender_submission.csv file size is", len(gender_submission_df))
```

    gender_submission.csv file size is 418


### 2.3.2 Granularity

The key is "PassengerId" -- the IDs of each passenger.   
 
According to the data, we can conclude that each line of the data contains the information of a passenger.

### 2.3.3 Scope


```python
survived_data = train_df['Survived']
print("Survived data scope:", survived_data.unique())
```

    Survived data scope: [0 1]



```python
class_data = train_df['Pclass']
print("Class data scope:", np.sort(class_data.unique()))
```

    Class data scope: [1 2 3]



```python
sex_data = train_df['Sex']
print("Sex data scope:", sex_data.unique())
```

    Sex data scope: ['male' 'female']



```python
age_data = train_df[train_df['Age'].notna()]
age_data_1 = age_data['Age'].unique()
print("Age data scope:", age_data_1.min(), '-', age_data_1.max())
```

    Age data scope: 0.42 - 80.0



```python
sibsp_data = train_df['SibSp']
print("Sibsp data scope:", np.sort(sibsp_data.unique()))
```

    Sibsp data scope: [0 1 2 3 4 5 8]



```python
parch_data = train_df['Parch']
print("Parch data scope:", np.sort(parch_data.unique()))
```

    Parch data scope: [0 1 2 3 4 5 6]



```python
fare_data = train_df[train_df['Fare'].notna()]
print(fare_data['Fare'].describe())
abs_fare = abs((fare_data['Fare'] - fare_data['Fare'].mean())/fare_data['Fare'].std())
fare_unusual = fare_data.loc[abs_fare>3]
print("\nFare data scope:", fare_data['Fare'].unique().min(), '-', fare_data['Fare'].unique().max())
print("\nUnusual fare data:", len(fare_unusual))
```

    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64
    
    Fare data scope: 0.0 - 512.3292
    
    Unusual fare data: 20



```python
fare_unusual
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>118</th>
      <td>119</td>
      <td>0</td>
      <td>1</td>
      <td>Baxter, Mr. Quigg Edmond</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
    </tr>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>299</th>
      <td>300</td>
      <td>1</td>
      <td>1</td>
      <td>Baxter, Mrs. James (Helene DeLaudeniere Chaput)</td>
      <td>female</td>
      <td>50.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
    </tr>
    <tr>
      <th>311</th>
      <td>312</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Emily Borie</td>
      <td>female</td>
      <td>18.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
    </tr>
    <tr>
      <th>341</th>
      <td>342</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>377</th>
      <td>378</td>
      <td>0</td>
      <td>1</td>
      <td>Widener, Mr. Harry Elkins</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>113503</td>
      <td>211.5000</td>
      <td>C82</td>
      <td>C</td>
    </tr>
    <tr>
      <th>380</th>
      <td>381</td>
      <td>1</td>
      <td>1</td>
      <td>Bidois, Miss. Rosalie</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>438</th>
      <td>439</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Mark</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>527</th>
      <td>528</td>
      <td>0</td>
      <td>1</td>
      <td>Farthing, Mr. John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17483</td>
      <td>221.7792</td>
      <td>C95</td>
      <td>S</td>
    </tr>
    <tr>
      <th>557</th>
      <td>558</td>
      <td>0</td>
      <td>1</td>
      <td>Robbins, Mr. Victor</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
    </tr>
    <tr>
      <th>689</th>
      <td>690</td>
      <td>1</td>
      <td>1</td>
      <td>Madill, Miss. Georgette Alexandra</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>1</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
    </tr>
    <tr>
      <th>700</th>
      <td>701</td>
      <td>1</td>
      <td>1</td>
      <td>Astor, Mrs. John Jacob (Madeleine Talmadge Force)</td>
      <td>female</td>
      <td>18.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>C62 C64</td>
      <td>C</td>
    </tr>
    <tr>
      <th>716</th>
      <td>717</td>
      <td>1</td>
      <td>1</td>
      <td>Endres, Miss. Caroline Louise</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>C45</td>
      <td>C</td>
    </tr>
    <tr>
      <th>730</th>
      <td>731</td>
      <td>1</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
    </tr>
    <tr>
      <th>742</th>
      <td>743</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Susan Parker "Suzette"</td>
      <td>female</td>
      <td>21.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
    </tr>
    <tr>
      <th>779</th>
      <td>780</td>
      <td>1</td>
      <td>1</td>
      <td>Robert, Mrs. Edward Scott (Elisabeth Walton Mc...</td>
      <td>female</td>
      <td>43.0</td>
      <td>0</td>
      <td>1</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B3</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
cabin_data = train_df['Cabin']
print("Unique Values:", len(cabin_data.unique()))
```

    Unique Values: 148



```python
embarked_data = train_df['Embarked']
print("Embarked data scope:", embarked_data.unique())
```

    Embarked data scope: ['S' 'C' 'Q' nan]


### 2.3.4 Temporality

In fact, the data in Titanic does not have much to do with timeliness, because no data in the table will be out of date. So we don't need to consider it in data analysis.

### 2.3.5 Faithfulness

Whether the data is reliable is very important. If the information is wrong, the result is meaningless.

**Name**

Check whether there are duplicate names in the table. If there are duplicate data, it needs to be deleted.



```python
name_data = train_df['Name']
if len(name_data)-len(name_data.unique()) == 0:
  print("No duplicate data in Name")
else:
  print("Duplicate data in Name")
```

    No duplicate data in Name


**Age**

There are some numbers in the age record that are not integers.


```python
age_unusual = age_data.loc[age_data['Age']*100%100>0]
print("\nUnusual Age data:", len(age_unusual))
```

    
    Unusual Age data: 25



```python
age_unusual
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57</th>
      <td>58</td>
      <td>0</td>
      <td>3</td>
      <td>Novel, Mr. Mansouer</td>
      <td>male</td>
      <td>28.50</td>
      <td>0</td>
      <td>0</td>
      <td>2697</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>78</th>
      <td>79</td>
      <td>1</td>
      <td>2</td>
      <td>Caldwell, Master. Alden Gates</td>
      <td>male</td>
      <td>0.83</td>
      <td>0</td>
      <td>2</td>
      <td>248738</td>
      <td>29.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>111</th>
      <td>112</td>
      <td>0</td>
      <td>3</td>
      <td>Zabour, Miss. Hileni</td>
      <td>female</td>
      <td>14.50</td>
      <td>1</td>
      <td>0</td>
      <td>2665</td>
      <td>14.4542</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>116</th>
      <td>117</td>
      <td>0</td>
      <td>3</td>
      <td>Connors, Mr. Patrick</td>
      <td>male</td>
      <td>70.50</td>
      <td>0</td>
      <td>0</td>
      <td>370369</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>122</th>
      <td>123</td>
      <td>0</td>
      <td>2</td>
      <td>Nasser, Mr. Nicholas</td>
      <td>male</td>
      <td>32.50</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>123</th>
      <td>124</td>
      <td>1</td>
      <td>2</td>
      <td>Webber, Miss. Susan</td>
      <td>female</td>
      <td>32.50</td>
      <td>0</td>
      <td>0</td>
      <td>27267</td>
      <td>13.0000</td>
      <td>E101</td>
      <td>S</td>
    </tr>
    <tr>
      <th>148</th>
      <td>149</td>
      <td>0</td>
      <td>2</td>
      <td>Navratil, Mr. Michel ("Louis M Hoffman")</td>
      <td>male</td>
      <td>36.50</td>
      <td>0</td>
      <td>2</td>
      <td>230080</td>
      <td>26.0000</td>
      <td>F2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>152</th>
      <td>153</td>
      <td>0</td>
      <td>3</td>
      <td>Meo, Mr. Alfonzo</td>
      <td>male</td>
      <td>55.50</td>
      <td>0</td>
      <td>0</td>
      <td>A.5. 11206</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>0</td>
      <td>3</td>
      <td>van Billiard, Mr. Austin Blyler</td>
      <td>male</td>
      <td>40.50</td>
      <td>0</td>
      <td>2</td>
      <td>A/5. 851</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>203</th>
      <td>204</td>
      <td>0</td>
      <td>3</td>
      <td>Youseff, Mr. Gerious</td>
      <td>male</td>
      <td>45.50</td>
      <td>0</td>
      <td>0</td>
      <td>2628</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>227</th>
      <td>228</td>
      <td>0</td>
      <td>3</td>
      <td>Lovell, Mr. John Hall ("Henry")</td>
      <td>male</td>
      <td>20.50</td>
      <td>0</td>
      <td>0</td>
      <td>A/5 21173</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>296</th>
      <td>297</td>
      <td>0</td>
      <td>3</td>
      <td>Hanna, Mr. Mansour</td>
      <td>male</td>
      <td>23.50</td>
      <td>0</td>
      <td>0</td>
      <td>2693</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>305</th>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.92</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
    </tr>
    <tr>
      <th>331</th>
      <td>332</td>
      <td>0</td>
      <td>1</td>
      <td>Partner, Mr. Austen</td>
      <td>male</td>
      <td>45.50</td>
      <td>0</td>
      <td>0</td>
      <td>113043</td>
      <td>28.5000</td>
      <td>C124</td>
      <td>S</td>
    </tr>
    <tr>
      <th>469</th>
      <td>470</td>
      <td>1</td>
      <td>3</td>
      <td>Baclini, Miss. Helene Barbara</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>2666</td>
      <td>19.2583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>525</th>
      <td>526</td>
      <td>0</td>
      <td>3</td>
      <td>Farrell, Mr. James</td>
      <td>male</td>
      <td>40.50</td>
      <td>0</td>
      <td>0</td>
      <td>367232</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>644</th>
      <td>645</td>
      <td>1</td>
      <td>3</td>
      <td>Baclini, Miss. Eugenie</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>2666</td>
      <td>19.2583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>676</th>
      <td>677</td>
      <td>0</td>
      <td>3</td>
      <td>Sawyer, Mr. Frederick Charles</td>
      <td>male</td>
      <td>24.50</td>
      <td>0</td>
      <td>0</td>
      <td>342826</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>735</th>
      <td>736</td>
      <td>0</td>
      <td>3</td>
      <td>Williams, Mr. Leslie</td>
      <td>male</td>
      <td>28.50</td>
      <td>0</td>
      <td>0</td>
      <td>54636</td>
      <td>16.1000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>755</th>
      <td>756</td>
      <td>1</td>
      <td>2</td>
      <td>Hamalainen, Master. Viljo</td>
      <td>male</td>
      <td>0.67</td>
      <td>1</td>
      <td>1</td>
      <td>250649</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>767</th>
      <td>768</td>
      <td>0</td>
      <td>3</td>
      <td>Mangan, Miss. Mary</td>
      <td>female</td>
      <td>30.50</td>
      <td>0</td>
      <td>0</td>
      <td>364850</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>803</th>
      <td>804</td>
      <td>1</td>
      <td>3</td>
      <td>Thomas, Master. Assad Alexander</td>
      <td>male</td>
      <td>0.42</td>
      <td>0</td>
      <td>1</td>
      <td>2625</td>
      <td>8.5167</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>814</th>
      <td>815</td>
      <td>0</td>
      <td>3</td>
      <td>Tomlin, Mr. Ernest Portage</td>
      <td>male</td>
      <td>30.50</td>
      <td>0</td>
      <td>0</td>
      <td>364499</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>831</th>
      <td>832</td>
      <td>1</td>
      <td>2</td>
      <td>Richards, Master. George Sibley</td>
      <td>male</td>
      <td>0.83</td>
      <td>1</td>
      <td>1</td>
      <td>29106</td>
      <td>18.7500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>843</th>
      <td>844</td>
      <td>0</td>
      <td>3</td>
      <td>Lemberopolous, Mr. Peter L</td>
      <td>male</td>
      <td>34.50</td>
      <td>0</td>
      <td>0</td>
      <td>2683</td>
      <td>6.4375</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



Other data cannot verify the correctness of the data. If recorder used paper to record, there may be some clerical errors. Or something else will happen, so we won’t consider it here for the time being.

## 2.4 Representative and Assumptions

**Representative**  
Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).

The data is obtained representative.  
Survival may be different in other passenger groups not included in the training sample.

**Assumptions**      

*   **Correlating**. each feature correlate with Surviva.
*   The survival situation of the personnel not included in the training data is consistent with that of the personnel in the training data.



## 2.5 Sampling Method

stratified sampling.

Stratified sampling is a probability sampling technique wherein the researcher divides the entire population into different subgroups or strata, then randomly selects the final subjects proportionally from the different strata.

For example, We can clearly find that gender and cabin are two very important factors, which can be treated as the strata.



# 3. Data Transformation and EDA 

## 3.1 Transformation


In stage1, we have converted the data into DataFrame. Now, we are going to convert some string to number and process some continuous data.


```python
combine = [train_df, test_df]
```

**Sex**

Let us start by converting Sex feature to a new format.
     {'female': 1, 'male': 0}


```python
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



**Age**

Age is a numerical continuous feature. However, we could easily find that Age contains missing or null values. We need to estimate and fill in these missing data.

Save the original age data in a new field *OriginAge*.


```python
train_df['OriginAge'] = train_df['Age']
```

*How to guess missing or null values?*

1. A simple way is to generate random numbers between mean and standard deviation.

2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...

Method 2 is better.


```python
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/axisgrid.py:316: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)





    <seaborn.axisgrid.FacetGrid at 0x7f7623c34cd0>




    
![png](output_74_2.png)
    



```python
guess_ages = np.zeros((2,3))
```


```python
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
```


```python

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# replace Age with ordinals based on these bands.
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

# remove the AgeBand feature.
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>OriginAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>



**Embarked**

Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.


```python
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
```




    'S'




```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>0.339009</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>OriginAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>1</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3.2 Features






```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>OriginAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>1</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>




- **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived. We decide to include this feature in our model.

 


```python
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.188908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.742038</td>
    </tr>
  </tbody>
</table>
</div>



- **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate.


```python
sns.kdeplot(train_df["OriginAge"][train_df.Survived == 1], color="blue", shade=True)
sns.kdeplot(train_df["OriginAge"][train_df.Survived == 0], color="navy", shade=True)

plt.xlim(0,85)

plt.legend(['Survived', 'Did not survived'])
plt.xlabel('Age range')
plt.ylabel('Rate')
plt.title('The relationship between age and survival')
```




    Text(0.5, 1.0, 'The relationship between age and survival')




    
![png](output_89_1.png)
    



```python
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived')

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.345395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.464286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.535885</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Parch</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.343658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.550847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



- **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features.


```python
plt.figure(figsize=(18,8))
sns.kdeplot(train_df["Fare"][train_df.Survived == 1], color="blue", shade=True)
sns.kdeplot(train_df["Fare"][train_df.Survived == 0], color="navy", shade=True)

plt.legend(['Survived', 'Did not survived'])
plt.xlabel('Fare range')
plt.ylabel('Rate')
plt.title('The relationship between fare and survival')
```




    Text(0.5, 1.0, 'The relationship between fare and survival')




    
![png](output_93_1.png)
    



```python
Embarked_survived = train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
Embarked_survived
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.339009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.389610</td>
    </tr>
  </tbody>
</table>
</div>



**Other features**

There are also some attributes we will use in the next, like name, etc.


## 3.3 Readiness of the data



Based on the previous processing, we don’t need to do more processing.  
We choose the attributes we discussed before to make predictions.  


```python
choose_column = ['Pclass','Sex', 'Age',"Embarked"]
```


```python
X_train = train_df[choose_column]
Y_train = train_df["Survived"]
X_test  = test_df[choose_column]
X_train.shape, Y_train.shape, X_test.shape
```




    ((891, 4), (891,), (418, 4))




```python
train_df[choose_column]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>



# 4. Modeling

## 4.1 Model

**Model: Decision Tree**  
The reason we chose the decision tree model is that it is the most common machine learning model, and we think it can meet our prediction goal.  
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.
Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.


```python
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()

```


```python
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)

```


```python
acc_decision_tree = decision_tree.score(X_train, Y_train)
acc_decision_tree
```




    0.8271604938271605



## 4.2 Model and Feature

**Currently, 4 features will use in our train model:**

* Pclass

* Sex

* Age

* Embarked 

The reason we choosen these 4 features in our train model is that these 4 attributes have very important meaning.

Later, we will consider the influence of other attributes, such as title, etc. 


```python
choose_column = ['Pclass','Sex', 'Age',"Embarked"]
```

## 4.3 Loss Function


---
**Comments**  
You have used decision tree as the model. If you are using sklearn, you should check its document and see what kind of loss function are used to do training.   


**<font color=red>Reply</font>**    
<font color=red>
If $\hat{y}_i$ is the predicted value of the i-th sample and $y_i$ is the corresponding true value, then the fraction of correct predictions over n is defined as

$$\Large
R(\theta) = \frac{1}{n} \sum_{i = 1}^n1( \hat{y}_i-y_i  )
$$
</font>.  

We will use MSE (mean square error) as our loss function.    

$$\Large
R(\theta) = \frac{1}{n} \sum_{i = 1}^n(y_i - \hat{y}_i )^2
$$



```python
def mean_squared_error(theta, data):
    n = data.size
    return sum((data-theta)**2) / n
```


```python
theta_values = np.linspace(0, 1, 100)
mse = [mean_squared_error(theta, train_df.Survived) for theta in theta_values]
plt.plot(theta_values, mse)
plt.xlabel(r'$\theta$')
plt.ylabel('L2 loss')
plt.title(r'L2 Loss for different values of $\theta$')
```




    Text(0.5, 1.0, 'L2 Loss for different values of $\\theta$')




    
![png](output_115_1.png)
    


**The implement** 


```python
def l2_survival_risk(theta):
  
    n = Y_pred.shape[0]
    return sum((theta-Y_pred)**2) / n
   
```

## 4.4 Estimate

`scipy.minimize` is a powerful method that can determine the optimal value of a variety of different functions. In practice, it is used to minimize functions that have no (or difficult to obtain) analytical solutions (it is a **numerical method**).


```python
from scipy.optimize import minimize


init_thetas = np.zeros(Y_pred.shape[0])
minimized = minimize(l2_survival_risk,init_thetas) 

minimized 
```




          fun: 1.2915465927964333e-17
     hess_inv: array([[1., 0., 0., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 1., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 1.]])
          jac: array([3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 7.03133564e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           7.03133564e-11, 7.03133564e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           7.03133564e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133575e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           7.03133564e-11, 7.03133564e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           7.03133564e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           1.01121567e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 1.01121567e-11,
           7.03133564e-11, 3.56487110e-11, 1.01121567e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           1.01121567e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 1.01121567e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133585e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 1.01121567e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           7.03133564e-11, 1.01121567e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 1.01121567e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 1.01121567e-11, 7.03133564e-11,
           1.01121567e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           7.03133564e-11, 1.01121567e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 1.01121567e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           1.01121567e-11, 7.03133564e-11, 3.56487110e-11, 1.01121567e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 7.03133564e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 7.03133564e-11,
           3.56487110e-11, 7.03133564e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11, 3.56487110e-11, 3.56487110e-11,
           7.03133564e-11, 3.56487110e-11, 7.03133564e-11, 7.03133564e-11,
           3.56487110e-11, 3.56487110e-11, 7.03133564e-11, 3.56487110e-11,
           3.56487110e-11, 3.56487110e-11])
      message: 'Optimization terminated successfully.'
         nfev: 2940
          nit: 4
         njev: 7
       status: 0
      success: True
            x: array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 1.00000001,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 1.00000001,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 1.00000001, 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.        , 0.        , 1.00000001, 1.00000001,
           1.00000001, 1.00000001, 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.        , 1.00000001, 0.        , 1.00000001,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 1.00000001, 1.00000001,
           0.        , 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 1.00000001, 1.00000001, 1.00000001,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.99999999, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.99999999, 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.99999999, 1.00000001, 0.        , 0.99999999, 1.00000001,
           0.        , 0.        , 0.99999999, 0.        , 1.00000001,
           0.        , 0.99999999, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.99999999, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.99999999, 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.99999999, 0.        ,
           0.        , 1.00000001, 0.        , 0.99999999, 0.        ,
           1.00000001, 0.        , 0.99999999, 0.        , 1.00000001,
           0.99999999, 0.        , 1.00000001, 0.        , 0.        ,
           0.99999999, 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.99999999, 1.00000001,
           0.99999999, 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.99999999, 0.        , 1.00000001, 0.99999999,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.99999999, 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.99999999, 1.00000001, 0.        ,
           0.99999999, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.00000001, 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 1.00000001, 0.        , 1.00000001,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 1.00000001, 0.        , 0.        ,
           1.00000001, 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           1.00000001, 1.00000001, 0.        , 0.        , 1.00000001,
           0.        , 0.        , 0.        ])




```python
theta_hat = minimized['x']
theta_hat
```




    array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 1.00000001,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 1.00000001,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 1.00000001, 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.        , 0.        , 1.00000001, 1.00000001,
           1.00000001, 1.00000001, 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.        , 1.00000001, 0.        , 1.00000001,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 1.00000001, 1.00000001,
           0.        , 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 1.00000001, 1.00000001, 1.00000001,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.99999999, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.99999999, 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.99999999, 1.00000001, 0.        , 0.99999999, 1.00000001,
           0.        , 0.        , 0.99999999, 0.        , 1.00000001,
           0.        , 0.99999999, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           0.        , 0.99999999, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.99999999, 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.99999999, 0.        ,
           0.        , 1.00000001, 0.        , 0.99999999, 0.        ,
           1.00000001, 0.        , 0.99999999, 0.        , 1.00000001,
           0.99999999, 0.        , 1.00000001, 0.        , 0.        ,
           0.99999999, 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.99999999, 1.00000001,
           0.99999999, 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.99999999, 0.        , 1.00000001, 0.99999999,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.99999999, 0.        ,
           0.        , 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.99999999, 1.00000001, 0.        ,
           0.99999999, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 1.00000001, 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 1.00000001, 0.        , 1.00000001,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 1.00000001,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 0.        , 0.        , 0.        ,
           0.        , 1.00000001, 0.        , 0.        , 0.        ,
           1.00000001, 0.        , 1.00000001, 0.        , 0.        ,
           1.00000001, 0.        , 1.00000001, 0.        , 0.        ,
           0.        , 0.        , 0.        , 1.00000001, 0.        ,
           1.00000001, 1.00000001, 0.        , 0.        , 1.00000001,
           0.        , 0.        , 0.        ])



## 4.5 Leaderboard position on Kaggle




**Generate submission file**


```python
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 2 columns</p>
</div>




```python
submission.to_csv('submission.csv', index=False)
```

**Submission**

![image.png](data:image/png;base64,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IEB+oON9quAPhMtmEUAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAgcUETjVOqwf54kMvAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQOBmERAgP9DxPlXwB8JlswgggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggMBiAqcap9WDfPGhlwACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCBwswgIkB/oeJ8q+APhslkEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEBgMYFTjdPqQb740EsAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQuFkEBMgPdLxPFfyBcNksAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAgggsJjAqcZp9SBffOglgAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCNwsAgLkBzrepwr+QLhsFgEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEFhM4FTjtHqQLz70EkAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQRuFgEB8gMd71MFfyBcNosAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggsJnCqcVo9yBcfegkggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACN4uAAPmBjvepgj8QLptFAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEFhM41TitHuSLD70EEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAgZtFQID8QMf7VMEfCJfNIoAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAosJnGqcVg/yxYdeAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggMDNIiBAfqDjfargD4TLZhFAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAIHFBE41TqsH+eJDLwEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEDgZhEQID/Q8T5V8AfCZbMIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIDAYgKnGqfVg3zx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f/3O/cjb3/L4B3/wBytuyWIECBAgQIAAAQIECBAgQIAAgf0JKJDvz+vA5p4q/IEBWBEBAgQIECBAgAABAsdaIPc8S2G8FsL7vn/hC1945Pd8P9bwdo4AAQIECBAgQIAAAQIECBBYKDDVOq0R5Au71R8JECBAgAABAgQIECCwfoGvfvWrs1e/+tWzhz3sYfPLqeey6qeffvrsWc961uz973//+hupBQQIECBAgAABAgQIECBAgMBWCSiQr6m7pwq/Ji6bJUCAAAECBAgQIEDgGAh84xvfmH3rW986BntiFwgQIECAAAECBAgQIECAAIGpCky1TmsE+VQTp90ECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBBYk4ACOfg1CdgsAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEjlZAgfxovedbmyr8fAd8Q4AAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgYkJTLVO6xLrEwua5hIgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGDdAgrka+qBqcKvictmCRAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgMFpgqnVaI8hHd70VECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAYLsEFMjX1N9ThV8Tl80SIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIEBgtMBU67RGkI/ueisgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIDAdgkokK+pv6cKvyYumyVAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMBoganWaY0gH931VkCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIHtElAgX1N/TxV+TVw2S4AAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgdECU63TGkE+uuutgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAtsloEC+pv6eKvyauGyWAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECowWmWqc1gnx011sBAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEtktAgXxN/T1V+DVx2SwBAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgRGC0y1TmsE+eiutwICBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAhsl4AC+Zr6e6rwa+KyWQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECIwWmGqd1gjy0V1vBQQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENguAQXyNfX3VOHXxGWzBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQGC0w1TqtEeSju94KCBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgsF0CCuRr6u+pwq+Jy2YJECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECAwWmCqdVojyEd3vRUQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIEBguwQUyNfU31OFXxOXzRIgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGC0wFTrtEaQj+76g1nBo//mZ2e+GMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMjAdmTgYKqM61uLAvma7KcK3+VyoNuOA51+1s8yIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAPJwNSnqdZpjSDfkOQ5EDoQyoAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyMD2ZGBDypQrN0OBfGW6cQtOFb671w5223Ow09f6WgZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkoFsvnNrPU63TGkG+IUlzEHQQlAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkIHtycCGlClXboYC+cp04xacKnx3rx3studgp6/1tQzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIQLdeOLWfp1qnNYJ8Q5LmIOggKAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAPbk4ENKVOu3AwF8pXpxi04VfjuXjvYbc/BTl/raxmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQgW69cGo/T7VOawT5hiTNQdBBUAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAa2JwMbUqZcuRkK5CvTjVtwqvDdvXaw256Dnb7W1zIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgA9164dR+nmqd1gjyDUmag6CDoAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAxsTwY2pEy5cjMUyFemG7fgVOG7e+1gtz0HO32tr2VABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABrr1wqn9PNU6rRHkG5I0B0EHQRmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARnYngxsSJly5WYokK9MN27BqcJ399rBbnsOdvpaX8uADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADHTrhVP7eap1WiPINyRpDoIOgjIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzKwPRnYkDLlys1QIF+ZbtyCU4Xv7rWD3fYc7PS1vpYBGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGejWC6f281TrtEaQb0jSHAQdBGVABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRgezKwIWXKlZuhQL4y3bgFpwrf3WsHu+052OlrfS0DMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMtCtF07t56nWaY0g35CkOQg6CMqADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMjA9mRgQ8qUKzdDgXxlunELThW+u9cOdttzsNPX+loGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZKBbL5zaz1Ot0xpBviFJcxB0EJQBGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZCB7cnAhpQpV26GAvnKdOMWnCp8d6+P+mD3qJfdanbX3/nx2c3u+v2zG9z6O2fXufF3zH7wlP8y+/Gf+67Zbe7/Q7P7PvmU2VG3aVO3d+fH3HB281/5nztfd/rtG3L5m+15YdrUTGqXDMqADMiADMiADMiADMiADMiADMiADMiADMiADMiADMjA9DPQrRdO7eep1mmNIN+QpB3lQexXfvdGOwXx773B/zNb9HXre1939lsvv/XWF4RP/bXrzJ1Ou+d1D8Xjfk85ZeekhJyYkJMXjjIPtjX9F1B9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I8g3JGlH/YRr2/utv7717DeefYvZfZ98yuzX/+Knd13Su82zzGNGUT/4WT+zs56s6yHPvsVJ9+JeZj0HPU8tWKdA3tafy5VnNPf9n/rTBzrSOg7N835POWXH5Ldevvue5K0NQ4+PfMlpswc+42Z79kUu1Z6Cf7yzrYc+99SNMB/aL7/3j4UMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMnMjAhpQpV26GAvnKdOMWnCp8d68dDE4cDA7SYqhAfpDbsK7D6TuuXGVABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABo5zBrr1wqn9PNU6rRHkG5K04/zkXue+KZB74Vxn/mxb/mRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABoYysCFlypWboUC+Mt24BacK393roSeG3487aCqQj/OTP34yIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMycDgZ6NYLp/bzVOu0RpBvSNIcWA7nwKJAfjiu8spVBmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABsZlYEPKlCs3Q4F8ZbpxC04VvrvXDiDjDiBDfre423Vmp9zp+3a+7vvkU2ZD8/n94fhz5SoDMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiAD/Rno1gun9vNU67RGkG9I0hwY+g8MXLjIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAwcxwxsSJly5WYokK9MN27BqcJ39/o4PqntkxcrGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGejPQLdeOLWfp1qnNYJ8Q5LmwNB/YODCRQZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZk4DhmYEPKlCs3Q4F8ZbpxC04VvrvXx/FJbZ+8WMmADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMhAfwa69cKp/TzVOq0R5BuSNAeG/gMDFy4yIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAPHMQMbUqZcuRkK5CvTjVtwqvDdvT6OT2r75MVKBmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABmRABvoz0K0XTu3nqdZpjSDfkKQ5MPQfGLhwkQEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZkAEZOI4Z2JAy5crNUCBfmW7cglOF7+71cXxS2ycvVjIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzLQn4FuvXBqP0+1TmsE+YYkzYGh/8DAhYsMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyMBxzMCGlClXboYC+cp04xacKnx3r4/jk9o+ebGSARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQARmQgf4MdOuFU/t5qnVaI8g3JGkODP0HBi5cZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGZEAGjmMGNqRMuXIzFMhXphu34FThu3t9HJ/U9smLlQzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAzIgAz0Z6BbL5zaz1Ot0xpBviFJc2DoPzBw4SIDMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMiADMnAcM7AhZcqVm6FAvjLduAWnCt/d6+P4pLZPXqxkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkoD8D3Xrh1H6eap3WCPINSZoDQ/+BgQsXGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZABGZCB45iBDSlTrtwMBfKV6cYtOFX47l4fxye1ffJiJQMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAP9GejWC6f281TrtEaQTy1p2kuAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIE1CyiQr6kDpgq/Ji6bJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAwGiBqdZpjSAf3fVWQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAge0SUCBfU39PFX5NXDZLgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACB0QJTrdMaQT66662AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAEC2yWgQL6m/p4q/Jq4bJYAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQKjBaZapzWCfHTXWwEBAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgS2S0CBfE39PVX4NXHZLAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBEYLTLVOawT56K63AgIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECGyXgAL5mvp7qvBr4rJZAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIjBaYap3WCPLRXW8FBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQ2C4BBfI19fdU4dfEZbMECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAYLTDVOq0R5KO73goIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECCwXQIK5Gvq76nCr4nLZgkQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIDBaYKp1WiPIR3e9FRAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGC7BBTI19TfU4VfE5fNEiBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAYLTAVOu0RpCP7norIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAwHYJKJCvqb+nCr8mLpslQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIDAaIGp1mmNIB/d9VZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACB7RJQIF9Tf08Vfk1cNkuAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIHRAlOt0xpBPrrrrYAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQLbJaBAvqb+nir8mrhslgABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAqMFplqnNYJ8dNdbAQECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBLZLQIF8Tf09Vfg1cdksAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIERgtMtU5rBPnorrcCAgSOu8CXr7p29q7zL5l945vfWnlXP/XFK2Yf+syXV17eggQIECBAgAABAgQIECBAgAABAgQIECBAgACBTRJQIF9Tb0wVflWu33vdB2YPevlZs9/5u/etugrLESCwD4HzL/nq7Kf+5I07X7f/qzP2seSJWV95zgXzdTzhjR868Qff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QIAAAQIECBAgQIDA5ggokK+pL6YKvyautW72o5/7yuw+Lz5z5+tfz/vCWttyFBv/4zd9ePagl581e/grz549++3nzS7/2tePYrMbt41t6/eN6wANIkCAAAECBAgQIECAAAECBAgQIECAAAECBAgcgsBU67RGkB9CGKySAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECx1lAgXxNvTtV+DVx2SwBAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgRGC0y1TmsE+eiutwICBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAhsl4AC+Zr6e6rwa+KyWQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECIwWmGqd1gjy0V1vBQQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENguAQXyNfX3VOHXxGWzBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQGC0w1TqtEeSju94KCBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgsF0CCuRr6u+pwq+Jy2YJECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECAwWmCqdVojyEd3vRUQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIEBguwQUyNfU31OFXxOXzRIgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGC0wFTrtEaQj+56KyBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMB2CSiQr6m/pwq/Ji6bJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAwGiBqdZpjSAf3fVWQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAge0SUCBfU39PFX5NXDZLgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACB0QJTrdMaQT66662AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAEC2yWgQL6m/p4q/Jq4bJYAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQKjBaZapzWCfHTXWwEBAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgS2S0CBfE39PVX4NXHZLAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBEYLTLVOawT56K63AgIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECGyXgAL5mvp7qvBr4rJZAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIjBaYap3WCPLRXW8FBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQ2C4BBfI19fdU4dfEZbMECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAYLTDVOq0R5KO73goIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECCwXQIK5Gvq76nCr4nLZgkQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIDBaYKp1WiPIR3e9FRAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGC7BBTI19TfU4VfE5fNEiBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAYLTAVOu0RpCP7norIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAwHYJKJCvqb+nCr8mLpslQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIDAaIGp1mmNIB/d9VZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACB7RJQIF9Tf08Vfk1cNkuAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIHRAlOt0xpBPrrrrYAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQLbJaBAvqb+nir8mrhslgABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAqMFplqnNYJ8dNdbAQECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBLZLQIF8Tf09Vfg1cdksAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIERgtMtU5rBPnorrcCAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIbJeAAvma+nuq8GvislkCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAiMFphqndYI8tFdbwUECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBDYLgEF8jX191Th18RlswQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIEBgtMNU6rRHko7veCggQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQILBdAgrka+rvqcKvictmCRAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgMFpgqnVaI8hHd70VECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAYLsEFMjX1N9ThV8Tl80SIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIEBgtMBU67RGkI/ueisgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIDAdgkokK+pv6cKvyYumyVAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMBoganWaY0gH931VkCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIHtElAgX1N/TxV+TVw2S4AAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgdECU63THpsR5N/61rdGd6IVECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMBigdRmFcgXGx3aXy/6zMU7+Ndce+2hbcOKCRAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQODbAqnNpkCeWu3UpsmPIL/00st28C/70pemZq+9BAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQmJxAarMpkKdWO7Vp8gXya665Zgc/HWAU+dTip70ECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECExJoI0e36nPXnPNlJq+09bJF8izF20U+ac+fYEi+eQiqMEECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECExBIMXx1GSnOno8xseiQJ4dueDCi+YjyTOkP52Tm8ObCBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGA1gdRcU3ttl1VPcTy12alOx6ZAng5oI8nTKb4YyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyIAMyMDBZmCK9x2vxfxjVSDPjuWe5OmUiz5zsSK5EwVkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkQAZkYGQGUntNDTa12KlPx65APvUO0X4CBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQOBwBBfLDcbVWAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENgwAQXyDesQzSFAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACBwxFQID8cV2slQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgQ0TUCDfsA7RHAIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBA4HAEF8sNxtVYCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQ2DABBfIN6xDNIUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIHDEVAgPxxXayVAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACBDRNQIN+wDtEcAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIEDgcAQXyw3G1VgIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBDYMAEF8g3rEM0hQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgcMRUCA/HFdrJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIENE1Ag37AO0RwCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQOBwBBfLDcbVWAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENgwAQXyDesQzSFAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACBwxFQID8cV2slQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgQ0TUCDfsA7RHAIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBA4HAEF8sNxtVYCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQ2DABBfIN6xDNIUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIHDEVAgPxxXayVAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACBDRNQIN+wDtEcAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIEDgcAQXyw3G1VgIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBDYMAEF8g3rEM0hQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgcMRUCA/HFdrJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIENE1Ag37AO0RwCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQOBwBBfLDcbVWAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENgwAQXyDesQzSFAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACBwxFQID8cV2slQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgQ0T+P8BtQRlsAL4+Q0AAAAASUVORK5CYII=)

**Leadernboard position**

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB9QAAAE8CAYAAACRhYrAAAAgAElEQVR4Aezd6XMc153me/8BPe/uvGm/mHtjZqK7J2YmxtEeh7t9vXS73b5qt6/Dy3WrvUq2ZC2WWtZuyiIlSqQorhIlihJ3UhRIcadIcd/3fd8XkCBBEDsIEABBUpTOjSfVp3jyVGZVoVAoVFV+EYHIWnL95ScThXrynPyCKbGfCxdrjH75oQJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAChSyAj03b5mOzq7Qb1f3DfPpZ59FLqazs9NcvXo1429tba3Zv3+/OXz4sLl9+3Ywn5MnT5rz588Hrx07dswsX77cjBs3zlRVVZl33nnHvP3228HzuXPnmo6OjshlX+j41Dy766b5lw23Qr8fXbgVOT4vUgEqQAWoABWgAlSgrxUgp42u4BeiXx64V9lRA1d7lkwFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAUqvQI3em6GAnUF7J3d3ebOnU8jN/3OnTtB6B0XrJ85c8Zs3rzZKFjv7u42N2/eNI2NjaapqcnU19cHj9etWxeE6WvXrg2C9CFDhpj169cH40ct9FjLHfObzeEgXcH620cJ06PqxWtUgApQASpABahAYSpAThtdRwL16LrwKhWgAlSAClABKkAFqAAVoAJUgApQASpABagAFaACFVqB7ohQXcH6rdufxG7xp59+arq6ukxzc3OqxXpdXZ3ZvXt3EKi3tLQYPW9tbQ2CdD1vb283p06dMkePHjUrV640e/fuNQcOHIgN0rXwnVdvh1qk2xbqIw8RpsfuHN6gAlSAClABKkAFClIBAvXoMhKoR9eFV6kAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVqOAKRLVUV6iu1+O6gLflUKv1GzdumGvXrgUB+YkTJ4KQXIF7Q0NDEKorUG9razPXr18PWq1r/Ew/WqaW3XCtyzy6LdzVOy3TM1WO96gAFaACVIAKUIFCVYBAPbqSBOrRdeFVKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAUqvAJR91S391i/eevz+6EXowRall2uhicaulOt1LlnejH2AMugAlSAClABKkAFVAEC9WgHBOrRdeFVKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAUSUAF18+6G2e7j613dRmH3Z599VvBKaJ6at5bhLtM+XnzuRtD9e8EXzAypABWgAlSAClABKhBTAQL16MIQqEfXhVepABWgAlSAClABKkAFqAAVoAJUgApQASpABagAFUhIBe58+qnpvtETGWzbgFvdsd/+5BPTl2hd02oecd3N22VpXbRO/FABKkAFqAAVoAJUoJgVIFCPrjaBenRdeJUKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQgYRV4Nbt+BbjNuzWsOtGj1F38QrH79z5NLjnuhu067Huia73NI7G1TTuPKIeq7W61oEfKkAFqAAVoAJUgAoMRAUI1KOrTqAeXRdepQJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlQggRX4vCv2W+Z6Z1fWADwqFM/nNS3r5q1b/dK1fAJ3IZtMBagAFaACVIAK5FkBAvXowhGoR9eFV6kAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVSHAFFKzfunXbdHbf6LdgXfPWMvrjHu0J3nVsOhWgAlSAClABKpBnBQjUowtHoB5dF16lAlSAClABKkAFqAAVoAJUgApQASpABagAFaACVIAKBBW4c+dO0G17IcJ1zUNdwGue/FABKkAFqAAVoAJUoJQqQKAevTcI1KPrwqtUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAClCBtAro3ui6L7q6aO/uuWm6um8Y3fvc7+pdr+k9jaNxNY2m5YcKUAEqQAWoABWgAqVaAQL16D1DoB5dF16lAlSAClABKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSACiSmAgTq0buaQD26LrxKBagAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQASpABagAFUhMBQjUo3c1gXp0XXiVClABKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAClABKpCYChCoR+9qAvXouvAqFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQASpABagAFaACVCAxFSBQj97VBOrRdeFVKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqEBiKkCgHr2rCdSj68KrVIAKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQASpA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)

# 5. Inference

## 5.1 Model
The two models are: **Decision Tree** and **RandomForestClassifier**.



**Decision Tree**:
Decision Tree has been described and applied in Project-2.


**RandomForestClassifier**: 
RandomForestClassifier is a random forest classifier.
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if `bootstrap=True` (default), otherwise the whole dataset is used to build each tree

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## 5.2 standard evaluation metric

Because decision trees and random forests belong to classifiers, we choose accuracy as the standard evaluation metric.



**accuracy_score**:  
Accuracy classification score.  
In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

```sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)[source]```

Reference:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html?highlight=ccuracy_score#sklearn.metrics.accuracy_score

## 5.3  Splitting


```python
from sklearn.model_selection import train_test_split 

train, test = train_test_split(train_df, test_size=0.25, random_state=100)

X_train = train[choose_column].values
Y_train = train["Survived"].values
X_test  = test[choose_column].values
Y_test = test["Survived"].values

print("Training Data Size: ", len(train))
print("Test Data Size: ", len(test))

```

    Training Data Size:  668
    Test Data Size:  223


## 5.4 Performance comparison


```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
'''
Args:
    modelA and modelB: sklearn models with fit and predict functions 
    X_train (data_frame): Data
    Y_train (data_frame): Label 

Return:
    Accuracy vector containing 5 accuracies for modelA
    Accuracy vector containing 5 accuracies for modelB
    the average accuracy for the 5 splits of modelA
    the average accuracy for the 5 splits of modelB
'''

def compute_CV_accuracy(modelA, modelB, X_train, Y_train):

    kf = KFold(n_splits=5)
    validation_accuracies_A = []
    validation_accuracies_B = []
    
    for train_idx, valid_idx in kf.split(X_train):
        # split the data
        split_X_train, split_X_valid =  X_train[train_idx],  X_train[valid_idx]
        split_Y_train, split_Y_valid =  Y_train[train_idx],  Y_train[valid_idx]

        # Fit the modelA on the training split
        modelA.fit(split_X_train, split_Y_train)
        
        # Compute the prediction accuracy on the validation split
        X_valid_pred = modelA.predict(split_X_valid)

        # accuracy
        accuracyA = accuracy_score(split_Y_valid, X_valid_pred.round(), normalize=True)
        validation_accuracies_A.append(accuracyA)

        # Fit the modelB on the training split
        modelB.fit(split_X_train, split_Y_train)
        X_valid_pred = modelB.predict(split_X_valid)
        
        # Compute the prediction accuracy on the validation split
        accuracyB = accuracy_score(split_Y_valid, X_valid_pred.round(), normalize=True)
        validation_accuracies_B.append(accuracyB)
        
    return validation_accuracies_A, np.mean(validation_accuracies_A), validation_accuracies_B, np.mean(validation_accuracies_B)
```


```python
from sklearn.ensemble import RandomForestClassifier

# Model
# decision_tree definited in part-2
modelA = decision_tree
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
modelB = rfc
# compute_CV_accuracy
validation_accuracies_A, mean_accuracies_A, validation_accuracies_B, mean_accuracies_B = compute_CV_accuracy(modelA, modelB, X_train, Y_train)

# average accuracy
print("the average accuracy of modelA(decision_tree): ",mean_accuracies_A)
print("the average accuracy of modelB(RandomForestClassifier): ",mean_accuracies_B)
winner = "decision_tree" if mean_accuracies_A > mean_accuracies_B  else "RandomForestClassifier"
print(winner,"is better!")
```

    the average accuracy of modelA(decision_tree):  0.8248344742453148
    the average accuracy of modelB(RandomForestClassifier):  0.8263270115587475
    RandomForestClassifier is better!


Through the comparison of accuracy score, we can conclude that RandomForestClassifier is better.


```python
from scipy import stats

# significance level
level = 0.05

statistic, pvalue = stats.ttest_ind(validation_accuracies_A, validation_accuracies_B)

# print("statistic:", statistic)
print("pvalue:", pvalue)

result = "pvalue is bigger than 0.05" if pvalue > level  else "pvalue is smaller than 0.05"
print(result)

```

    pvalue: 0.9117551386481355
    pvalue is bigger than 0.05


we could find that the pvalue(0.9117551386481355) is more than the significance level 0.05. Therefore, for this case, we could not believe that one is significantly better than another. 

## 5.5 submission

**Generate submission file**


```python
y = train_df["Survived"]
features = choose_column
X = train_df[choose_column]
X_test = test_df[choose_column]

model = rfc
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.read_csv('data/gender_submission.csv')
submission['Survived'] = predictions
submission.to_csv('another_submission.csv', index=False)
```

**new leaderboard position**

![image.png](data:image/png;base64,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)

**old leaderboard position**

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB9QAAAE8CAYAAACRhYrAAAAgAElEQVR4Aezd6XMc153me/8BPe/uvGm/mHtjZqK7J2YmxtEeh7t9vXS73b5qt6/Dy3WrvUq2ZC2WWtZuyiIlSqQorhIlihJ3UhRIcadIcd/3fd8XkCBBEDsIEABBUpTOjSfVp3jyVGZVoVAoVFV+EYHIWnL95ScThXrynPyCKbGfCxdrjH75oQJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAChSyAj03b5mOzq7Qb1f3DfPpZ59FLqazs9NcvXo1429tba3Zv3+/OXz4sLl9+3Ywn5MnT5rz588Hrx07dswsX77cjBs3zlRVVZl33nnHvP3228HzuXPnmo6OjshlX+j41Dy766b5lw23Qr8fXbgVOT4vUgEqQAWoABWgAlSgrxUgp42u4BeiXx64V9lRA1d7lkwFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAUqvQI3em6GAnUF7J3d3ebOnU8jN/3OnTtB6B0XrJ85c8Zs3rzZKFjv7u42N2/eNI2NjaapqcnU19cHj9etWxeE6WvXrg2C9CFDhpj169cH40ct9FjLHfObzeEgXcH620cJ06PqxWtUgApQASpABahAYSpAThtdRwL16LrwKhWgAlSAClABKkAFqAAVoAJUgApQASpABagAFaACFVqB7ohQXcH6rdufxG7xp59+arq6ukxzc3OqxXpdXZ3ZvXt3EKi3tLQYPW9tbQ2CdD1vb283p06dMkePHjUrV640e/fuNQcOHIgN0rXwnVdvh1qk2xbqIw8RpsfuHN6gAlSAClABKkAFClIBAvXoMhKoR9eFV6kAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVqOAKRLVUV6iu1+O6gLflUKv1GzdumGvXrgUB+YkTJ4KQXIF7Q0NDEKorUG9razPXr18PWq1r/Ew/WqaW3XCtyzy6LdzVOy3TM1WO96gAFaACVIAKUIFCVYBAPbqSBOrRdeFVKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAUqvAJR91S391i/eevz+6EXowRall2uhicaulOt1LlnejH2AMugAlSAClABKkAFVAEC9WgHBOrRdeFVKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAUSUAF18+6G2e7j613dRmH3Z599VvBKaJ6at5bhLtM+XnzuRtD9e8EXzAypABWgAlSAClABKhBTAQL16MIQqEfXhVepABWgAlSAClABKkAFqAAVoAJUgApQASpABagAFUhIBe58+qnpvtETGWzbgFvdsd/+5BPTl2hd02oecd3N22VpXbRO/FABKkAFqAAVoAJUoJgVIFCPrjaBenRdeJUKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQgYRV4Nbt+BbjNuzWsOtGj1F38QrH79z5NLjnuhu067Huia73NI7G1TTuPKIeq7W61oEfKkAFqAAVoAJUgAoMRAUI1KOrTqAeXRdepQJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlQggRX4vCv2W+Z6Z1fWADwqFM/nNS3r5q1b/dK1fAJ3IZtMBagAFaACVIAK5FkBAvXowhGoR9eFV6kAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVSHAFFKzfunXbdHbf6LdgXfPWMvrjHu0J3nVsOhWgAlSAClABKpBnBQjUowtHoB5dF16lAlSAClABKkAFqAAVoAJUgApQASpABagAFaACVIAKBBW4c+dO0G17IcJ1zUNdwGue/FABKkAFqAAVoAJUoJQqQKAevTcI1KPrwqtUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAClCBtAro3ui6L7q6aO/uuWm6um8Y3fvc7+pdr+k9jaNxNY2m5YcKUAEqQAWoABWgAqVaAQL16D1DoB5dF16lAlSAClABKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSACiSmAgTq0buaQD26LrxKBagAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQASpABagAFUhMBQjUo3c1gXp0XXiVClABKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAClABKpCYChCoR+9qAvXouvAqFaACVIAKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQASpABagAFaACVCAxFSBQj97VBOrRdeFVKkAFqAAVoAJUgApQASpABagAFaACVIAKUAEqQAWoABWgAlSAClABKkAFqEBiKkCgHr2rCdSj68KrVIAKUAEqQAWoABWgAlSAClABKkAFqAAVoAJUgApQASpA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)

The effect of the two submissions looks the same.

# 6. Conclusion and Learning

## 6.1 Conclusion

The accuracy of our model reaches 82.5%, which proves the important role of data analysis in prediction. Nevertheless, our model still has a lot of room for optimization. Because of time, we have not achieved higher accuracy. We will continue to deepen in the future.

## 6.2 Learning

Nowadays, the application of big data is more and more extensive, it provides more valuable information for every industry. This time through the analysis of the survival rate of the Titanic, I have a deeper understanding of the big data analysis process. In the initial stage, it is necessary to do the data wrangling to ensure that the content of the research is meaningful. Furthermore, it is very important to clean and filter the data before researching the data, which can reduce the deviation. Then I found the rules by constructing the model, learned the use of important functions such as loss function, and how to use A/B test to choose the best solution. This project allowed me to consolidate the content of the entire assignments and gained a lot.*----Deyi*

Through this research project, I came into contact with gaggle, which is an interesting platform, and learned how to deal with problems from my peers. My interest in data analysis has increased.
*----Guanchen*
