# Databricks notebook source
# MAGIC %md #ANALYSIS OF ACCIDENTS DATA AROUND US

# COMMAND ----------

# MAGIC %md ##Step1 - Loading the dataset, creating a schema and checking the data-types of each attribute

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/accidents.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
accidents_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# COMMAND ----------

# MAGIC %md ####Displaying the top 1000 rows of the dataset

# COMMAND ----------

display(accidents_df)

# COMMAND ----------

# MAGIC %md ####Creating a table so we can run SQL commands

# COMMAND ----------

# Create a view or table

accidents_table = "accidents_csv"

accidents_df.createOrReplaceTempView(accidents_table)

# COMMAND ----------

# MAGIC %md ####Using SQL to check the contents of the table

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `accidents_csv`

# COMMAND ----------

# MAGIC %md ####Storing the dataframe into cache memory to increase the speed

# COMMAND ----------

accidents_df.cache()

# COMMAND ----------

# MAGIC %md ##Step2 - Performing the Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md ####Converting the dataframe to pandas and checking the statistics of each attribute

# COMMAND ----------

##descriptive statistics#
accidents_df.describe().toPandas().transpose()

# COMMAND ----------

# MAGIC %md ####Importing all the necessary pyspark libraries

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import dayofmonth,hour,dayofyear,weekofyear,month,year,format_number,date_format,mean, date_format, datediff, to_date, lit
import math
from pyspark.sql.functions import mean as _mean, stddev as stddev, col
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
from pyspark.sql.functions import unix_timestamp, from_unixtime, date_format
from pyspark.sql.functions import date_format
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
plt.style.use('fivethirtyeight')
from pyspark.sql.functions import isnan, when, count, col

# COMMAND ----------

# MAGIC %md ####Creating a pandas dataframe to perform the analysis

# COMMAND ----------

#creating a pandas dataframe
accidents_pd = accidents_df.toPandas()

# COMMAND ----------

# MAGIC %md ####Printing all the important details from the data - Number of rows, Number of attributes, Attribute names, Checking missing values and Unique vales per attribute 

# COMMAND ----------

print('Rows     :',accidents_df.count())
print('Columns  :',len(accidents_df.columns))
print('\nFeatures :\n     :',accidents_df.columns)
print('\nMissing values    :',accidents_pd.isnull().values.sum())
print('\nUnique values :  \n',accidents_pd.nunique())

# COMMAND ----------

# MAGIC %md ####Storing the attributes which has data-type as object

# COMMAND ----------

#collecting all the attributes with object datatypes
accidents_pd.select_dtypes(exclude=['int','float']).columns

# COMMAND ----------

# MAGIC %md ####Checking first few entries of 'Description' column

# COMMAND ----------

#First few rows from description column
accidents_pd['Description'].head()

# COMMAND ----------

# MAGIC %md ####Working with unique values to determine the actual source of data and number of time-zones considered

# COMMAND ----------

print(accidents_pd['Source'].unique())
print(accidents_pd['Description'].unique())
print(accidents_pd['Timezone'].unique())
print(accidents_pd['Amenity'].unique())

# COMMAND ----------

# MAGIC %md ##Step3- Analyzing the data visually to solve business questions

# COMMAND ----------

# MAGIC %md ###Business Question 1- Is there any relationship among the attributes, especially severity?

# COMMAND ----------

# MAGIC %md ######factors affecting the severity

# COMMAND ----------

#Creating a heatmap to find the correlation among each attribute contributing to the accident
fig=sns.heatmap(accidents_pd[['TMC','Severity','Start_Lat','End_Lat','Distance(mi)','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)']].corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
display()

# COMMAND ----------

# MAGIC %md ###We are focusing on the severity here
# MAGIC The range is from 1 to 4 where:
# MAGIC 1. Indicates the accident did not have much impact on the traffic (No delay as such)
# MAGIC 2. Indicates the accident had a minor impact on the traffic (Short delay)
# MAGIC 3. Indicates the accident had moderate impact on the traafic (Moderate delay)
# MAGIC 4. Indicates the accident had a very high impact on the traffic (High delay and clutter)

# COMMAND ----------

#creating a bar graph and a pie-chart to demonstrate percentage distribution of the severity
f,ax=plt.subplots(1,2,figsize=(18,8))
accidents_pd['Severity'].value_counts().plot.pie(explode=None,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Percentage Severity Distribution')
ax[0].set_ylabel('Count')
sns.countplot('Severity',data=accidents_pd,ax=ax[1],order=accidents_pd['Severity'].value_counts().index)
ax[1].set_title('Count of Severity')
plt.show()
display()

# COMMAND ----------

# MAGIC %md ###Business Question 2 - Which states contribute the most to the accidents data?

# COMMAND ----------

# MAGIC %md #####Visually analysing the percentage distribution of the number of accidents based on timezones

# COMMAND ----------

#creating a bar graph and a pie-chart to demonstrate percentage distribution of the number of accidents based on timezone
f,ax=plt.subplots(1,2,figsize=(18,8))
accidents_pd['Timezone'].value_counts().plot.pie(explode=None,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Accidents in Different Timezone')
ax[0].set_ylabel('Count')
sns.countplot('Timezone',data=accidents_pd,ax=ax[1],order=accidents_pd['Timezone'].value_counts().index)
ax[1].set_title('Accident Count Based on Timezone')
plt.show()
display()

# COMMAND ----------

# MAGIC %md #####Top 10 states prone to accidents based on the data 

# COMMAND ----------

#Creating a bar graph and a pi-chart to see the percentage distribution of the
from pyspark.sql.functions import sum as fsum
fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
accidents_pd.State.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title("Top 10 Accident Prone States",size=20)
ax[0].set_xlabel('States',size=18)
count=accidents_pd['State'].value_counts()
groups=list(accidents_pd['State'].value_counts().index)[:10]
counts=list(count[:10])
counts.append(count.agg('sum')-count[:10].agg('sum'))
groups.append('Other')
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ioff()
plt.ylabel('')
display()

# COMMAND ----------

# MAGIC %md ###Business Question 3 - Does weather really have an impact on accidents?

# COMMAND ----------

#Creating a bar graph based on the weather patterns during the accidents
fig, ax=plt.subplots(figsize=(16,7))
accidents_pd['Weather_Condition'].value_counts().sort_values(ascending=False).head(5).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('5 Top Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()
display()

# COMMAND ----------

#Converting the dates in the dataset to proper year-month-day format
accidents_pd['time'] = pd.to_datetime(accidents_pd.Start_Time, format='%Y-%m-%d %H:%M:%S')
accidents_pd = accidents_pd.set_index('time')

# COMMAND ----------

# MAGIC %md ###Business Question 4 - Which time of the year or week contribute more towards the accident?  

# COMMAND ----------

#Plotting scatter plots for the number of number accidents for daily, weekly and yearly
freq_text = {'D':'Daily','W':'Weekly','Y':'Yearly'}
for i, (fr,text) in enumerate(freq_text.items(),1):
  sample = accidents_pd.ID['2016':].resample(fr).count()
  sample.plot(style='.')
  plt.title('Accidents, {} count'.format(text))
  plt.xlabel('Date')
  plt.ylabel('Accident Count')
  display()

# COMMAND ----------

# MAGIC %md #####Checking for the number of accidents that happen day-wise

# COMMAND ----------

#converting all the variables to their required data types to analyse
accidents_pd['Start_Time'] = pd.to_datetime(accidents_pd['Start_Time'], format="%Y/%m/%d %H:%M:%S")
accidents_pd['DayOfWeekNum'] = accidents_pd['Start_Time'].dt.dayofweek
accidents_pd['DayOfWeek'] = accidents_pd['Start_Time'].dt.weekday_name
accidents_pd['MonthDayNum'] = accidents_pd['Start_Time'].dt.day
accidents_pd['HourOfDay'] = accidents_pd['Start_Time'].dt.hour

# COMMAND ----------

#plotting a bar-graph for the number of accidents on different days of week
fig, ax=plt.subplots(figsize=(16,7))
accidents_pd['DayOfWeek'].value_counts(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Day of the Week',fontsize=20)
plt.ylabel('Number of accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Accident on Different Days of Week',fontsize=25)
plt.grid()
plt.ioff()
display()
