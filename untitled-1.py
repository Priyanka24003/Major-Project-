# -*- coding: utf-8 -*-
"""


## Dataset Information

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.


Variable | Description
----------|--------------
Item_Identifier | Unique product ID
Item_Weight | Weight of product
Item_Fat_Content | Whether the product is low fat or not
Item_Visibility | The % of total display area of all products in a    store allocated to the particular product
Item_Type | The category to which the product belongs
Item_MRP | Maximum Retail Price (list price) of the product
Outlet_Identifier | Unique store ID
Outlet_Establishment_Year | The year in which store was established
Outlet_Size | The size of the store in terms of ground area covered
Outlet_Location_Type | The type of city in which the store is located
Outlet_Type | Whether the outlet is just a grocery store or some sort of supermarket
Item_Outlet_Sales | Sales of the product in the particulat store. This is the outcome variable to be predicted."""

## Import modules


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# %matplotlib inline
warnings.filterwarnings('ignore')

"""## Loading the dataset"""

df = pd.read_csv('Train.csv')
df.head(50)

# statistical info
df.describe()

# datatype of attributes
df.info()

# check unique values in dataset
df.apply(lambda x: len(x.unique()))
# count of unique value

"""## Preprocessing the dataset"""

# check for null values
df.isnull().sum()
# check if na none or empty

df1 = df[df['Item_Weight'].isna()]
df1

# check for categorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col







# name class section

cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col

df.value_counts()

# print the categorical columns
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()


#     har ek category ke kitne value present he , column wise

# fill the missing values
item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
item_weight_mean

miss_bool = df['Item_Weight'].isnull().value_counts()
miss_bool



for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])

df['Item_Weight'].isnull().sum()

outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
outlet_size_mode

miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

df['Outlet_Size'].isnull().sum()

sum(df['Item_Visibility']==0)

# replace zeros with mean
df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)

sum(df['Item_Visibility']==0)

# combine item fat content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()

"""## Creation of New Attributes"""

df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type']

df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df['New_Item_Type'].value_counts()

df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()

# create small values for establishment year
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

df['Outlet_Years']

df.head()

"""## Exploratory Data Analysis"""

sns.distplot(df['Item_Weight'])

sns.distplot(df['Item_Visibility'])

sns.distplot(df['Item_MRP'])

sns.distplot(df['Item_Outlet_Sales'])

# log transformation
df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])

sns.distplot(df['Item_Outlet_Sales'])

sns.countplot(df["Item_Fat_Content"])

# plt.figure(figsize=(15,5))
l = list(df['Item_Type'].unique())
chart = sns.countplot(df["Item_Type"])
chart.set_xticklabels(labels=l, rotation=90)

sns.countplot(df['Outlet_Establishment_Year'])

sns.countplot(df['Outlet_Size'])

sns.countplot(df['Outlet_Location_Type'])

sns.countplot(df['Outlet_Type'])

"""## Coorelation Matrix


"""

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

df.head()
