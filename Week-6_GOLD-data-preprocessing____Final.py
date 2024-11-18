#!/usr/bin/env python
# coding: utf-8

# <html> <h1 style="font-style:bold; color:blue;"> Machine Learning in Finance </h1> </html>

# <html> <h1 style="font-style:italic; color:blue;"> Week-6 </h1> </html>

# <html> <h2 style="font-style:italic; color:blue;"> Financial Data Preprocessing </h2> </html>

# In[ ]:





# ### Financial data of GOLD during 2022
# 
# ### High & Low prices are outputs
# - Data preprocessing done
# - Spreads of all prices (Open, High, Low, Close) between Ask and Bid are calculated
# - Delta of trading volumes (as well as the modulus of this delta) between Ask and Bid volumes
# are introduced as additional parameters
#  - Starts of each trading day and each trading week are marked
# 
# 
# - normalization parameters are calculated only for the training part (80%) of the Dataset (so as not to peep into the future)
# 
# #### Number of rows
# - 354.694 rows
# 
# 

# ![image.png](attachment:3a946d34-9ee0-4974-968f-1750321d9754.png)

# In[ ]:


# Please run the next 4 cells. 
# After each cell completed, please restart the Kernel

# It is needed to run only one time for the computer


# In[2]:


pip install --upgrade pip


# In[1]:


pip install plotly


# In[1]:


pip install cufflinks


# In[1]:


pip install --upgrade mplfinance


# ___________________________![image.png](attachment:02538446-b908-4d47-88ff-21f37151e181.png)

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import *
import seaborn as sns


# In[1]:





# _____________________________![image.png](attachment:2c02e05f-c4fc-4fa0-9a02-447d4c166047.png)

# In[3]:


df_2022_ask = pd.read_csv('XAUUSD_1 Min_Ask_2022.01.01_2022.12.31.csv')
df_2022_bid = pd.read_csv('XAUUSD_1 Min_Bid_2022.01.01_2022.12.31.csv')


# In[2]:





# ________________________![image.png](attachment:dff6c63b-26b7-47d9-9049-cc57c0bbc99f.png)

# In[4]:


print(df_2022_ask.head(3))
print(df_2022_ask.tail(3))


# In[5]:





# _____________________![image.png](attachment:fc4d1a07-10ff-46f1-82d9-d8132f01ed5c.png)

# In[5]:


print(df_2022_bid.head(3))
print(df_2022_bid.tail(3))


# In[7]:





# In[9]:


# rows numbers and 'Time (UTC)' are the same in the rows with the same indices for Ask and Bid datasets


# ___________________________![image.png](attachment:8d14f7ac-e4f6-43a2-ab06-79e45a37426b.png)

# In[6]:


df_2022_ask.info()


# In[11]:





# ________________________![image.png](attachment:c21cd255-253c-411d-87e9-cafc4c28d000.png)

# In[7]:


df_2022_bid.info()


# In[13]:





# In[ ]:





# <html> <h3 style="font-style:italic; color:blue;"> Merge DataFrame </h3> </html>

# In[272]:


#df_2022_Bid.merge?


# ________________________![image.png](attachment:839546c2-5b52-4466-a2e0-c852ce625459.png)

# In[9]:


df_2022 = df_2022_bid.merge(df_2022_ask, left_on='Time (UTC)', right_on='Time (UTC)', how='outer')
df_2022


# In[16]:





# ______________________________![image.png](attachment:50d24299-da79-4894-baf1-70e9eba7ba79.png)

# In[10]:


df_2022.info()


# In[18]:





# In[129]:


# No NaN elements ! It's Good !


# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[11]:


# rename columns

df_2022.columns = ['Local time', 'Open_Bid', 'High_Bid', 'Low_Bid', 'Close_Bid', 'Volume_Bid', 
                   'Open_Ask', 'High_Ask', 'Low_Ask', 'Close_Ask', 'Volume_Ask']


# ____________________________![image.png](attachment:72dca975-9582-472a-9a14-7fc96d5c848a.png)

# In[12]:


df_2022.head(3)


# In[22]:





# In[24]:


# Save preliminary dataset


# ________________________________![image.png](attachment:ede3be22-6068-4d86-b9e2-feac14e82ce5.png)

# In[13]:


file_obj2 = open('df_2022.csv', 'w')
df_2022.to_csv('df_2022.csv', encoding='utf-8', index=False)
file_obj2.close()


# In[25]:





# In[ ]:


# Please find the df_2022.csv file in your folder


# ___________________________![image.png](attachment:3d864952-3434-4cd9-89b6-bbbaf8b9c076.png)

# In[14]:


df_2022 = []
df_2022


# In[26]:





# In[ ]:





# In[31]:


# Upload the preliminary dataset


# _______________________![image.png](attachment:858484b4-ca57-4dd2-b4c4-b0d89683f78b.png)

# In[15]:


df = pd.read_csv('df_2022.csv', low_memory = False, sep=',')


# In[33]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next 2 cells.

# In[16]:


# Delta of trading volumes (as well as the modulus of this delta) between Ask 
# and Bid volumes are added as additional parameters

df["Volume_Delta"] = df["Volume_Ask"] - df["Volume_Bid"]
df["Volume_Delta_abs"] = (df["Volume_Ask"] - df["Volume_Bid"]).abs()


# In[17]:


df["Open_Delta"] = df["Open_Ask"]  - df["Open_Bid"]
df["High_Delta"] = df["High_Ask"]  - df["High_Bid"]
df["Low_Delta"]  = df["Low_Ask"]   - df["Low_Bid"]
df["Close_Delta"]= df["Close_Ask"] - df["Close_Bid"]


# ___________________________![image.png](attachment:bd380350-bc2b-40cc-bf6e-12b485b3de9e.png)

# In[18]:


df.describe()


# In[39]:





# ____________________________![image.png](attachment:6b5a4f2b-477f-45be-b57f-cde5336cba81.png)

# In[19]:


data = df.drop(['Open_Ask', 'High_Ask', 'Low_Ask', 'Close_Ask'], axis=1)


# In[41]:





# __________________________![image.png](attachment:f21af777-43e5-4e33-8fb8-9b9e1302bf39.png)

# In[20]:


data.shape


# In[43]:





# _________________________![image.png](attachment:3729f3d3-c49b-45f7-bf46-b26a7d4afd5e.png)

# In[21]:


data.head(3)


# In[45]:





# In[ ]:





# <html> <h3 style="font-style:italic; color:blue;"> Date transformation </h3> </html>

# ______________________![image.png](attachment:dae6143b-e91d-4718-a99b-41f540e145c4.png)

# In[22]:


import datetime


# In[53]:





# _________________________![image.png](attachment:332dee11-1c7c-4bd1-a167-dba9ae0c5c30.png)

# In[24]:


data['Local_time_T'] = pd.to_datetime(data['Local time'], utc=True)


# In[55]:





# ________________________![image.png](attachment:158a02d1-59c6-45bc-849f-40afb0732012.png)

# In[25]:


data=data.drop(['Local time'], axis=1)


# In[57]:





# ________________________![image.png](attachment:03c00080-3ec6-4b2d-92ab-ac274bdf48aa.png)

# In[26]:


data.info()


# In[59]:





# ____________________________![image.png](attachment:c7886931-5a68-4ca7-b354-be5ddecfa5b3.png)

# In[27]:


data.head(3)


# In[61]:





# In[ ]:





# <html> <h3 style="font-style:italic; color:blue;"> Data Visualisation </h3> </html>

# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[28]:


plt.figure(figsize=(12,6))
plt.plot(data['Close_Bid'])
plt.title('GOLD close price, 2022')
plt.xlabel('N minutes')
plt.ylabel('Close price')
plt.show()


# #### Candlestick chart for 70 minutes 

# _____________________________![image.png](attachment:7d5c4194-1388-4838-8c28-3e2e6b968ee0.png)

# In[29]:


data_chart = data.set_index('Local_time_T', inplace=False)


# In[67]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[30]:


data_chart = data_chart.drop(['Volume_Ask', 'Volume_Delta', 'Volume_Delta_abs', 
                        'Open_Delta', 'High_Delta', 'Low_Delta', 'Close_Delta'],axis=1)

# 


# ____________________________![image.png](attachment:e1cd0ccc-1818-4ec5-a54f-c6a5e7bafd0a.png)

# In[32]:


import mplfinance as mpf

data_chart.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
mpf.plot(data_chart.iloc[0:70], type='candle', style='charles', title='GOLD Candlestick Chart', volume=True)


# In[71]:





# _______________________________________![image.png](attachment:61e9f065-3f52-4745-a926-4a94fd4b86f0.png)

# In[33]:


data_chart = []
data_chart


# In[73]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[34]:


plt.figure(num=1,figsize=(12,5))
plt.hist(data['Close_Bid'],bins=100)
plt.title('Close (Bid) price distribution',size=18)
plt.ylabel('Numbers',size=14)
plt.xlabel('Close (Bid) price',size=14);


# In[ ]:





# <html> <h4 style="color:blue;"> To speed up writing the code in the following few cells that plot graphs, you can copy and then slightly modify the code from the previous cells. </h4> </html>
# 

# ____________________________![image.png](attachment:b3a9a999-6b00-4780-8e13-0685ee006e58.png)

# In[35]:


plt.figure(num=1,figsize=(10,5))
data['Close_Bid'].plot.kde()
plt.title('Close (Bid) price Density',size=18)
plt.ylabel('Density',size=14)
plt.xlabel('Close (Bid) price',size=14);


# In[78]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[36]:


fig = plt.figure(figsize=(12, 5))
plt.rc('axes', titlesize= 30 ) 
sns.set_style('whitegrid')
sns.set_context(rc={'legend.fontsize': 20.0}) 

sns.displot( 
            data[['High_Bid', 'Low_Bid']],
            height=8,
            aspect=1.7,
            #hue="species", 
            kde=True,
            element="step",
            bins=100,
            alpha=0.03,          
                        
)

plt.title('High & Low (Bid) Price Distribution')
plt.xlabel('Price', size= 20)
#plt.ylabel('count (%)')
plt.ylabel('Numbers', size= 20)
plt.show();


# __________________________________![image.png](attachment:bdd167f9-4eaa-4dd1-8294-dc8aa62645a7.png)

# In[37]:


plt.figure(num=1,figsize=(12,5))
plt.hist(data['Volume_Bid'],bins=100)
plt.title('Volume (Bid) price distribution',size=18)
plt.ylabel('Numbers',size=14)
plt.xlabel('Volume (Bid) price',size=14);


# In[82]:





# _____________________________![image.png](attachment:6d3eb2d0-499a-42c3-9505-4e6a83aadbd9.png)

# In[38]:


plt.figure(num=1,figsize=(12,5))
plt.hist(data['Volume_Ask'],bins=100)
plt.title('Volume (Ask) distribution',size=18)
plt.ylabel('Numbers',size=14)
plt.xlabel('Volume (Ask) price',size=14);


# In[ ]:





# In[84]:





# In[86]:


# Pay attention that large Bid volumes are larger than large Ask volumes.
# That is, trades are more significant when falling. This is a typical situation in financial markets.


# In[ ]:





# <html> <h3 style="color:blue;"> Print the thin long tail of the volume(Bid and Ask) histograms. </h3> </html>

# _________________________________![image.png](attachment:a4ca3f70-d897-4072-ab06-2e320326bd7c.png)

# In[40]:


vol_350k = data[ (data['Volume_Bid'] > 0.35) | (data['Volume_Ask'] > 0.35)]


# In[90]:





# ____________________________![image.png](attachment:1dd2db35-a1fd-4e9f-8d86-de3a523cc3ca.png)

# In[43]:


plt.figure(num=1,figsize=(15,7))
plt.hist(vol_350k[['Volume_Bid', 'Volume_Ask']],bins=100)
plt.legend(['Volume Bid', 'Volume Ask'])
plt.title('Volume > 350,000 distribution',size=18)
plt.ylabel('Numbers',size=14)
plt.xlabel('Volume',size=14);


# In[92]:





# ____________________________![image.png](attachment:82f5ee6a-0bc5-4f34-a00c-7f01ddba8499.png)

# In[44]:


vol_350k.plot.scatter(x='Volume_Bid', y='Volume_Ask');


# In[94]:





# In[96]:


# The same Distribution but with another style


# ![image.png](attachment:57ef4052-891d-4435-a8a9-df0ed5ffb2f5.png)

# In[46]:


import cufflinks as cf
cf.go_offline()


# In[98]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[47]:


vol_350k[['Volume_Ask', 'Volume_Bid']].iplot(
                            kind='hist',
                            histnorm='percent',
                            barmode='overlay',
                            xTitle='Volume > 350,000 distribution',
                            yTitle='Numbers',
                            title='Volume')


# In[102]:


# ! Click "Export to plot.ly" in the lower right corner


# <html> <h4 style="color:blue;"> To speed up writing the code in the following few cells that plot graphs, you can copy and then slightly modify the code from the previous cells. </h4> </html>

# _______________________![image.png](attachment:954b9dbf-1db5-4692-a092-e4d9ce565974.png)

# In[48]:


data[['High_Delta']].iplot(
                            kind='hist',
                            histnorm='percent',
                            barmode='overlay',
                            xTitle='High Delta price distribution',
                            yTitle='Numbers',
                            title='High delta price')


# In[104]:





# In[106]:


# ! Click "Export to plot.ly" in the lower right corner


# ![image.png](attachment:a7e408a1-b3a8-45a8-9386-efc9c7e2867b.png)

# In[49]:


data[['Low_Delta']].iplot(
                            kind='hist',
                            histnorm='percent',
                            barmode='overlay',
                            xTitle='Low delta price distribution',
                            yTitle='Numbers',
                            title='low delta price')


# In[108]:





# In[110]:


# ! Click "Export to plot.ly" in the lower right corner


# _________________________![image.png](attachment:88ef472f-0585-48b1-a7fe-22ac7d3be1ac.png)

# In[51]:


plt.figure(num=1,figsize=(15,7))
plt.hist(data[['Open_Delta']],bins=100)
plt.title('Open delta price distribution',size=18)
plt.ylabel('Numbers',size=13)
plt.xlabel('open delta price',size=13);


# In[112]:





# _____________________________![image.png](attachment:07a11bf8-164f-4565-9a40-abf4b617942d.png)

# In[52]:


plt.figure(num=1,figsize=(15,7))
plt.hist(data[['Close_Delta']],bins=100)
plt.title('close delta price distribution',size=18)
plt.ylabel('Numbers',size=13)
plt.xlabel('close delta price',size=13);


# In[114]:





# In[118]:


# Pay attention to the difference in the Delta distributions above !
# Clouse_Delta price could be significantly higher than other Delta prices,
# It means, that at the end of the trading day, we have a big spread that can destroy potential profit.


# ____________________________![image.png](attachment:4657f03e-b26a-499a-a16b-8d56e1f504a8.png)

# In[53]:


plt.figure(num=1,figsize=(12,5))
plt.hist(data[['Volume_Delta']],bins=100)
plt.title('Volume delta distribution',size=18)
plt.ylabel('Numbers',size=13)
plt.xlabel('Volume delta',size=13);


# In[122]:





# _________________________________![image.png](attachment:324a6544-75d3-4185-a12b-624838fd3e94.png)

# In[54]:


plt.figure(num=1,figsize=(12,5))
plt.hist(data[['Volume_Delta_abs']],bins=100)
plt.title('Volume delta abs distribution',size=18)
plt.ylabel('Numbers',size=13)
plt.xlabel('Volume delta abs',size=13);


# In[124]:





# _____________________________![image.png](attachment:a1947c8a-39b5-4495-b821-6bae6c9e7338.png)

# In[55]:


data[['Open_Delta', 'High_Delta', 'Low_Delta', 'Close_Delta']].iplot(
                                                                    kind = 'box',
                                                                    yTitle = 'Delta',
                                                                    title = 'Deltas')


# In[126]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[56]:


# plot the part of the whole dataset using iplot()
# The time period equals 400 minutes: from 19800 to 20200

data.iloc[19800:20200,:][['High_Bid', 'Low_Bid', 'Local_time_T', 'Volume_Ask','Volume_Bid']].iplot(
                                                x='Local_time_T', y=['High_Bid', 'Low_Bid'], 
                                                mode='lines+markers', 
                                                xTitle='Date', yTitle='Price',
                                                title='GOLD ')


# In[130]:


# ! Click "Export to plot.ly" in the lower right corner


# In[ ]:





# <html> <h3 style="font-style:italic; color:blue;"> Indicate starts of days and weeks </h3> </html>

# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[57]:


# Create a new time column by moving it down one row

data['Local_time_T_shipt_1_Down'] = data['Local_time_T'].shift(1)


# ____________________________![image.png](attachment:35db35dd-ef0e-450d-8531-0554536f2efa.png)

# In[58]:


data["Local_time_T_Delta"] = data['Local_time_T'] - data['Local_time_T_shipt_1_Down']


# In[134]:





# _______________________________![image.png](attachment:ae0a2c85-0e93-4548-9ebc-7b1d7b49ac8b.png)

# In[59]:


data.head(-5)


# In[138]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[60]:


# The start of the year is the start of a week
# Therefore, we change the time interval to 2 days (the duration of the weekend).

data['Local_time_T_Delta'].iloc[0] = "2 days 01:01:00"
data['Local_time_T_Delta'].iloc[0]


# ____________![image.png](attachment:aae0992c-067c-4a84-b148-c961ea97f6f7.png)

# In[61]:


data[['Local_time_T_Delta']][data['Local_time_T_Delta'] > '0 days 00:01:00'].value_counts(sort=False)


# In[142]:





# In[144]:


# We found the duration of night intervals when there is no trading in Gold.
# Total such intervals - 258, which is equal to the number of trading days in 2022 year


# _______________![image.png](attachment:b887d8a6-8300-4ab9-8b52-2d7f1a2b2aa7.png)

# In[63]:


new_day = data[['Local_time_T_Delta']][data['Local_time_T_Delta'] > '0 days 00:01:00']


# In[146]:





# _________________![image.png](attachment:eba5488f-36f7-4e7b-8c19-858dda10ebd6.png)

# In[64]:


new_day['Local_time_T_Delta'].value_counts(sort=False)


# In[148]:





# _____________________________![image.png](attachment:d4d34ba4-3379-4b99-94c1-9ef91ffda421.png)

# In[65]:


data[['Local_time_T_Delta']][data['Local_time_T_Delta'] > '1 days 00:00:00'].value_counts(sort=False)


# In[150]:





# In[152]:


# We found the weekend duration when there was no trading in Gold.
# Total intervals - 52, which is equal to the number of weekends in the 2022 year


# ____________________________![image.png](attachment:8a88acc4-2bde-4b16-a3f4-7ad5f2ce8af5.png)

# In[66]:


new_week = data[data['Local_time_T_Delta'] > '1 days 00:00:00']


# In[154]:





# _______________________![image.png](attachment:9f80cfca-3880-4d00-b254-4c17f2fe26c5.png)

# In[67]:


new_week['Local_time_T_Delta'].value_counts(sort=False)


# In[156]:





# ____________________![image.png](attachment:a07f2403-2a77-4911-b0e3-cf515299ee0d.png)

# In[68]:


data['New_day'] = 0
data['New_week'] = 0


# In[158]:





# _______________________________![image.png](attachment:f58f135b-ca06-4750-8b17-030ab16e2613.png)

# In[69]:


data.loc[data['Local_time_T_Delta'] > '0 days 00:01:00', 'New_day'] = 1


# In[160]:





# _________________________![image.png](attachment:1296cfe4-cf3f-425f-a4d1-b80455c7315e.png)

# In[70]:


data[data['New_day'] == 1]


# In[162]:





# __________________________________![image.png](attachment:e41cda67-10c1-49c8-ada5-996e2ddcb0a4.png)

# In[72]:


data.loc[data['Local_time_T_Delta'] > '1 days 00:00:00', 'New_week'] = 1


# In[164]:





# __________________________![image.png](attachment:32e015ed-c891-4f0c-b832-a730ce42f8f3.png)

# In[73]:


data[data['New_week'] == 1]


# In[166]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[74]:


# drop the columns that are no longer needed, 
# since the time series is regular 
# (each line is the next minute, except for the marked lines in the New_day and New_week columns

data2 = data.drop(['Local_time_T', 'Local_time_T_shipt_1_Down', 'Local_time_T_Delta'],axis=1)


# _________________![image.png](attachment:886f4198-302c-476b-95a6-5ca4be4882f2.png)

# In[75]:


data2.head(3)


# In[170]:





# _______________![image.png](attachment:7f5df341-e678-4411-89b6-7275fb04a9db.png)

# In[76]:


data2.info()


# In[172]:





# In[ ]:





# <html> <h3 style="font-style:italic; color:blue;"> Create Outputs (vector of answers) </h3> </html>

# 
# - we will predict 'High_Bid' and 'Low_Ask'
# since we are interested in the maximum and minimum prices of the next minute at which we can
#  - sell at the maximum price (High_Bid) or 
#  - buy at the minimum price (Low_Ask = Low_Bid + Low_Delta)
# 

# ____________________________![image.png](attachment:1ef6b82f-e63e-4144-b482-c2ef8cd6537d.png)

# In[77]:


data2['Y_High_Bid'] = data2['High_Bid']
data2['Y_Low_Ask'] = data2['Low_Bid'] + data2['Low_Delta']


# In[174]:





# _______________________![image.png](attachment:a82e5951-3635-4315-839b-3f6b377eed0d.png)

# In[78]:


data2.tail()


# In[176]:





# In[ ]:





# <html> <h2 style="font-style:italic; color:blue;"> Data Normalisation </h2> </html>

# Note:
# A typical mistake when choosing a part of the data and time for scaling is to scale the entire dataset before it is divided into test and training data. It is a mistake because scaling starts the calculation of statistics, that is, minima/maxima of variables. When realizing time series forecasting in real life, at the time of their generation, you cannot have information from future observations. Therefore, statistics should be calculated on the training data, and then the result obtained will be applied to the test data. To take information “from the future” to generate predictions (that is, from a test sample), the model will produce forecasts with “system bias

# In[178]:


# normalisation
# data=(data-data.min())/(data.max()-data.min())


# ![image.png](attachment:3d6c0c33-510a-4886-b86f-b3880ec31997.png)

# #### Create the TRAIN dataset to find Min and Max ​​for normalisation

# ##### Train / Validation / Test Split
# - ~80% for training
# - ~10% for validation
# - ~10% for testing

# In[183]:


# here we divide only the training set(80%) for the calculation of normalisation parameters
# The division into validation and training sets will be done immediately before training


# ______________________![image.png](attachment:18854c5f-7d1a-481f-89a9-652de3b20ac2.png)

# In[79]:


data_length = len(data)
data_length


# In[178]:





# _____________![image.png](attachment:9ab61505-abc2-40f8-8aa2-2435473b892c.png)

# In[82]:


train_size = int(round(data_length*0.8, -3))
train_size


# In[180]:





# _______________________![image.png](attachment:c23fd7a1-ca08-4b40-882b-2cc434826c1b.png)

# In[83]:


train = data2.iloc[:train_size]
train.shape


# In[182]:





# ______________________![image.png](attachment:5054bc1a-7846-470c-907c-eedbc5a97e5f.png)

# In[84]:


train.tail(2)


# In[184]:





# <html> <h4 style="font-style:italic; color:blue;"> Find Max and Min for Prices, Volumes and Deltas </h4> </html>

# In[194]:


# Max price is Max High_Ask = High_Bid + High_Delta
# Min price is Min Low_Bid

# Max Volume = Max{Volume_Bid, Volume_Ask}
# Min Volume = 0

# Max_Delta = Max{Open_Delta, High_Delta, Low_Delta, Close_Delta}
# Min_Delta = Min{Open_Delta, High_Delta, Low_Delta, Close_Delta}


# _________________![image.png](attachment:1ea5a270-e09a-48dd-9c02-33bd0c5e2916.png)

# In[85]:


max_price = (train['High_Bid'] + train['High_Delta']).max()
max_price


# In[186]:





# _______________![image.png](attachment:cae6134e-360a-41d9-a2e1-b96abbad7bb6.png)

# In[86]:


min_price = train['Low_Bid'].min()
min_price


# In[188]:





# _____________________![image.png](attachment:6776d572-ce61-42d3-bf23-2b469e99dc9c.png)

# In[87]:


max_volume = max(max(train['Volume_Bid']), max(train['Volume_Ask']))
max_volume


# In[ ]:





# In[190]:





# _____________________![image.png](attachment:b6975058-82f6-47f1-9e43-770459995fca.png)

# In[91]:


max_delta = max(max(train['Open_Delta']), max(train['High_Delta']), max(train['Low_Delta']), max(train['Close_Delta']))
max_delta


# In[192]:





# ______________________![image.png](attachment:010565cf-9ec6-48e8-8853-2cc8ff3be4d6.png)

# In[93]:


max_delta = round(max_delta, 3)
max_delta


# In[194]:





# ______________________![image.png](attachment:4e7297ac-91e8-4673-a41a-930a969463d7.png)

# In[94]:


min_delta = min(min(train['Open_Delta']), min(train['High_Delta']), min(train['Low_Delta']), min(train['Close_Delta']))
min_delta


# In[196]:





# ________________________![image.png](attachment:fd03cc5a-bb1b-4a82-9ebe-b0ed48c51cfa.png)

# In[95]:


min_delta = round(min_delta,3)
min_delta


# In[198]:





# _____________________![image.png](attachment:bfefeeb3-88c6-4fcc-adfb-ad7c197146b7.png)

# In[96]:


max_volume_delta = train['Volume_Delta'].max()
max_volume_delta


# In[200]:





# __________________________![image.png](attachment:8d2afcca-be68-416d-b727-cd4864cbaf4b.png)

# In[97]:


min_volume_delta = train['Volume_Delta'].min()
min_volume_delta


# In[202]:





# __________________![image.png](attachment:4bdb980c-c7cc-4e06-965b-ca3e4625c122.png)

# In[98]:


max_volume_delta_abs = train['Volume_Delta_abs'].max()
max_volume_delta_abs


# In[204]:





# ________________________![image.png](attachment:b9578a9b-dd40-42ba-a896-230a607ed16c.png)

# In[99]:


min_volume_delta_abs = train['Volume_Delta_abs'].min()
min_volume_delta_abs


# In[206]:





# In[218]:


# min_volume_Delta_abs = 0
# Therefore, the normalization formula for volume_Delta_abs is simplified


# <html> <h4 style="font-style:italic; color:blue;"> Normalise </h4> </html>
# Run each cell only one time !
# 

# #### Now just press 'CTRL' and 'ENTER' keys on the next 5 cells with code.

# In[100]:


data2['Open_Bid'] = ( data2['Open_Bid'] - min_price ) / (max_price-min_price)
data2['High_Bid']  = ( data2['High_Bid']  - min_price ) / (max_price-min_price)
data2['Low_Bid']   = ( data2['Low_Bid']   - min_price ) / (max_price-min_price)
data2['Close_Bid'] = ( data2['Close_Bid'] - min_price ) / (max_price-min_price)
data2['Y_High_Bid'] = ( data2['Y_High_Bid'] - min_price ) / (max_price-min_price)
data2['Y_Low_Ask']  = ( data2['Y_Low_Ask']  - min_price ) / (max_price-min_price)


# In[101]:


data2['Volume_Ask']  = data2['Volume_Ask'] / max_volume
data2['Volume_Bid']  = data2['Volume_Bid'] / max_volume


# In[104]:


data2['Volume_Delta'] = ( data2['Volume_Delta'] - min_volume_delta ) / (max_volume_delta-min_volume_delta)


# In[105]:


data2['Volume_Delta_abs']  = data2['Volume_Delta_abs'] / max_volume_delta_abs


# In[229]:


# For the price deltas, the minimum difference is more interesting, therefore we apply inverse normalisation
# data= 1 - (data - data.min()) / (data.max() - data.min()) = (data.max() - data) / (data.max() - data.min())


# In[107]:


data2['Open_Delta']  = ( max_delta - data2['Open_Delta'] )  / (max_delta-min_delta)
data2['High_Delta']  = ( max_delta - data2['High_Delta'] )  / (max_delta-min_delta)
data2['Low_Delta']   = ( max_delta - data2['Low_Delta'] )   / (max_delta-min_delta)
data2['Close_Delta'] = ( max_delta - data2['Close_Delta'] ) / (max_delta-min_delta)


# ________________________![image.png](attachment:87782c04-416b-47b9-9086-4a64b3239474.png)

# In[108]:


data2.head()


# In[220]:





# _____________________________![image.png](attachment:6735a6d9-23dc-4697-b456-c878f084cff5.png)

# In[109]:


data2.info()


# In[222]:





# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[110]:


# memory size reduction

columns_float =['Open_Bid', 'High_Bid', 'Low_Bid', 'Close_Bid', 
                'Volume_Bid', 'Volume_Ask', 'Volume_Delta', 'Volume_Delta_abs', 
                'Open_Delta', 'High_Delta', 'Low_Delta', 'Close_Delta',
                'Y_High_Bid', 'Y_Low_Ask']

columns_integer =['New_day', 'New_week']


# __________________________![image.png](attachment:c708961d-9a9d-4a86-89fe-9019a56248ca.png)

# In[111]:


for column in columns_float:
    data2[column] = pd.to_numeric(data2[column], downcast='float')
for column in columns_integer:
    data2[column] = pd.to_numeric(data2[column], downcast='integer')
    
data2.info()


# In[226]:





# In[ ]:





# ## <font color='green'>Save final dataset to file !</font>

# #### Now just press 'CTRL' and 'ENTER' keys on the next cell.

# In[112]:


# Do it once!
# Writing a normalised dataset to disk in file GOLD_2020_normilised.csv

file_obj1 = open('GOLD_2022_normilised.csv', 'w')
data2.to_csv('GOLD_2022_normilised.csv', encoding='utf-8', index=False)
file_obj1.close()


# In[ ]:





# # Lab Logbook Requirement:

# <html> <h3 style="font-style:italic; color:blue;">
#    
# 1) Plot the price chart of the part of the whole dataset 'High_Bid' and 'Low_Bid' prices using iplot() library.
# 2) The start point should equal the 5 last digits of your SID Number.
# 3) The time period (in minutes) should equal the 3 last digits of your SID Number.
# 4) Please only add a print-screen of your code and final graph to your Lab Logbook.
# </h3> </html>

# <html> <h3 style="color:red;">
# NOTE: DON'T FORGET TO SAVE AND BACK UP YOUR COMPLETED JUPYTER NOTEBOOK AND LAB LOGBOOK ON GITHUB OR ONEDRIVE.
# </h3> </html>

# In[114]:


data.iloc[33662:34323,:][['High_Bid', 'Low_Bid', 'Local_time_T']].iplot(
                                                    x='Local_time_T', y=['High_Bid', 'Low_Bid'], 
                                                    mode='lines+markers', 
                                                    xTitle='Date', yTitle='Price',
                                                    title='GOLD ')

