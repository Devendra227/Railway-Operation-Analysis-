#!/usr/bin/env python
# coding: utf-8

# # Railway Information  Analysis

# # Level 1: Data Exploration and Basic Operations

# ## Task 1.1: Load and Inspect Data

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import folium
import warnings 
warnings.filterwarnings


# In[3]:


data=pd.read_csv("D:\\data science\\Railway_info.csv")
print(data.head(10))


# In[8]:


data.columns


# In[4]:


print("\nData types of each column:")
print(data.dtypes)


# ###  Data Exploration

# In[43]:


print(data.info())
print(data.describe())


# In[5]:


print("\nMissing values in the dataset:")
print(data.isnull().sum())


# ## Task 1.2: Basic Statistics

# ### Calculate key statistics including:

# In[9]:


total_trains = data.shape[0]
unique_source_stations = data['Source_Station_Name'].nunique()
unique_destination_stations = data['Destination_Station_Name'].nunique()

print(f"\nTotal number of trains: {total_trains}")
print(f"Count of unique source stations: {unique_source_stations}")
print(f"Count of unique destination stations: {unique_destination_stations}")


# ###  Identify the most common source station

# In[10]:


most_common_source = data['Source_Station_Name'].mode()[0]
most_common_source_count = data['Source_Station_Name'].value_counts().iloc[0]

# Identify the most common destination station
most_common_destination = data['Destination_Station_Name'].mode()[0]
most_common_destination_count = data['Destination_Station_Name'].value_counts().iloc[0]

# Display the results
print(f"Most common source station: {most_common_source} (Count: {most_common_source_count})")
print(f"Most common destination station: {most_common_destination} (Count: {most_common_destination_count})")


# ## Task 1.3: Data Cleaning

# In[11]:


print("Missing values in each column:")
print(data.isnull().sum())


# In[12]:


data['Source_Station_Name'] = data['Source_Station_Name'].str.upper()
data['Destination_Station_Name'] = data['Destination_Station_Name'].str.upper()

# Display the first few rows to verify the changes
print("First 10 rows after standardizing station names:")
print(data.head(10))


# # Level 2: Data Transformation and Aggregation

# ## Task 2.1: Data Filtering

# In[14]:


# Filter for trains operating on Saturdays
# Make sure to use the exact string matching the day names in your dataset
saturdays_data= data[data['days'].str.contains('Saturday', case=False, na=False)]

# Display the first few rows of the filtered dataset
print("Trains operating on Saturdays:")
print(saturdays_data.head(10))


# In[18]:


station_of_interest = 'CST-MUMBAI'
# Filter the dataframe for trains starting from the specified station
trains_from_station_data = data[data['Source_Station_Name'] == station_of_interest]

# Display the first few rows of the new dataframe
print(f"Trains starting from {station_of_interest}:")
print(trains_from_station_data.head(10))


# ## Task 2.2: Grouping and Aggregation

# In[19]:


# Group by source station and compute the number of trains originating from each station
trains_per_station = data.groupby('Source_Station_Name').size().reset_index(name='Number_of_Trains')

# Sort the result by number of trains in descending order
trains_per_station_sorted = trains_per_station.sort_values(by='Number_of_Trains', ascending=False)

# Display the result
print("Number of trains originating from each station:")
print(trains_per_station_sorted)


# In[21]:


# Group by source station and day, and count the number of trains
trains_per_day_per_station = data.groupby(['Source_Station_Name', 'days']).size().reset_index(name='Number_of_Trains')

# Verify the content of the grouped dataframe
print("\nGrouped data (trains_per_day_per_station):")
print(trains_per_day_per_station.head())

# Compute the average number of trains per day for each source station
average_trains_per_day = trains_per_day_per_station.groupby('Source_Station_Name')['Number_of_Trains'].mean().reset_index(name='Average_Trains_Per_Day')

# Verify the content of the average dataframe
print("\nAverage trains per day (average_trains_per_day):")
print(average_trains_per_day.head())


# ## Task 2.3: Data Enrichment

# In[22]:


data['days'] = data['days'].astype(str)

# Define function to categorize days
def categorize_day(day):
    # List of weekdays and weekends
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    if day in weekdays:
        return 'Weekday'
    elif day in weekends:
        return 'Weekend'
    else:
        return 'Unknown'  # In case there are days not covered

# Apply the function to create a new column
data['Day_Category'] = data['days'].apply(categorize_day)

# Display the first few rows to verify the changes
print("First 10 rows after adding 'Day_Category' column:")
print(data.head(10))


# # Level 3: Advanced Data Analysis

# ## Task 3.1: Pattern Analysis

# In[23]:


import matplotlib.pyplot as plt

journey_distribution = data['days'].value_counts().sort_index()

# Plot the distribution
plt.figure(figsize=(10, 6))
journey_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Train Journeys Throughout the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Journeys')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()


# In[24]:


import matplotlib.pyplot as plt

# Count the number of trains by source station
source_station_counts = data['Source_Station_Name'].value_counts()

# Count the number of trains by destination station
destination_station_counts = data['Destination_Station_Name'].value_counts()

# Display the counts
print("Train counts by source station:")
print(source_station_counts.head())

print("\nTrain counts by destination station:")
print(destination_station_counts.head())


# In[25]:


plt.figure(figsize=(12, 6))
source_station_counts.head(10).plot(kind='bar', color='skyblue')
plt.title('Number of Trains by Source Station')
plt.xlabel('Source Station')
plt.ylabel('Number of Trains')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[26]:


plt.figure(figsize=(12, 6))
destination_station_counts.head(10).plot(kind='bar', color='salmon')
plt.title('Number of Trains by Destination Station')
plt.xlabel('Destination Station')
plt.ylabel('Number of Trains')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[28]:


import seaborn as sns

# Create a pivot table for heatmap
pivot_table = data.pivot_table(index='Source_Station_Name', columns='Destination_Station_Name', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=False, fmt='d')
plt.title('Train Movements Heatmap')
plt.xlabel('Destination Station')
plt.ylabel('Source Station')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ## Task 3.2: Correlation and Insights

# In[30]:


data['days'] = data['days'].astype(str)

# Count the number of trains for each day of the week
trains_per_day = data['days'].value_counts().sort_index()

# Display the aggregated data
print("Number of trains for each day of the week:")
print(trains_per_day)


# In[31]:


import matplotlib.pyplot as plt

# Bar Plot of Train Counts by Day
plt.figure(figsize=(10, 6))
trains_per_day.plot(kind='bar', color='lightblue')
plt.title('Number of Trains by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Trains')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()


# In[34]:


day_to_num = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

data['day_num'] = data['days'].map(day_to_num)

# Aggregate the number of trains per day number
trains_per_day_num = data.groupby('day_num').size()

# Sort by day number
trains_per_day_num = trains_per_day_num.sort_index()

# Compute the correlation coefficient
correlation = np.corrcoef(list(day_to_num.values()), trains_per_day_num)[0, 1]

print(f"\nCorrelation between day of the week and number of trains: {correlation:.2f}")


# ## Develop insights and recommendations based on the analysis to inform decisionmaking.

# # Insights
# Distribution of Train Journeys Throughout the Week:
# 
# From the bar plot showing the number of trains for each day, identify the days with the highest and lowest number of train journeys.
# ### Source and Destination Station Trends:
# 
# Identify which stations have the highest number of departures and arrivals.
# Recognize any major hubs or popular destinations.
# 
# ### Correlation Analysis:
# 
# Determine if there is any significant correlation between the days of the week and the number of train journeys.
# Assess if certain days consistently have more or fewer trains.
# Recommendations
# Based on the insights derived from the data, here are some recommendations:
# 
# ### Optimize Train Schedules:
# 
# High Demand Days: Increase the number of trains on days with higher demand to accommodate passenger flow.
# Low Demand Days: Reduce the number of trains or deploy smaller trains on days with lower demand to optimize resource utilization.
# 
# ### Focus on Major Hubs:
# 
# Improvement in Services: Enhance facilities and services at major hubs identified by high train counts to improve passenger experience.
# 
# Infrastructure Development: Invest in infrastructure improvements at these stations to handle the high traffic efficiently.
# Promote Balanced Distribution:
# 
# Incentivize Travel on Low-Demand Days: Offer discounts or promotions for travel on days with lower train counts to balance the load across the week.
# 
# #### Flexible Scheduling
# Implement flexible scheduling to dynamically adjust train frequencies based on real-time demand data.
# Strategic Planning for New Routes:
# 
# #### Identify Potential New Routes:
# Analyze under-served routes or connections between high-traffic source and destination stations to introduce new train services.
# 
# #### Customer Feedback Integration: 
# Incorporate passenger feedback to identify desired routes or services that are currently lacking.
# Operational Efficiency:
# 
# #### Resource Allocation:
# Allocate resources such as staff and maintenance more effectively by aligning with the days and stations with the highest demand.
# 
# #### Maintenance Scheduling:
# Plan maintenance activities on days with lower demand to minimize disruption to passengers.
# Implementation Plan
# 
# ### To implement these recommendations, consider the following steps:
# 
# ### Data-Driven Decision Making:
# 
# Establish a data analytics team to continuously monitor and analyze train operations data.
# Use advanced analytics and machine learning models to predict demand and optimize schedules.
# 
# ### Passenger Engagement:
# 
# Conduct surveys and gather feedback from passengers to understand their preferences and pain points.
# Use this feedback to make informed decisions about train services and amenities.
# 
# ### Collaboration with Stakeholders:
# 
# Work closely with local governments, transportation authorities, and other stakeholders to align train services with broader transportation and urban planning strategies.
# 
# ### Technology Integration:
# 
# Implement advanced ticketing and scheduling systems that can dynamically adjust based on demand.
# Use IoT and real-time monitoring to track train movements and passenger flow.
# Visualization and Reporting
# 
# Regularly visualize and report the findings to stakeholders through dashboards and detailed reports. Use tools like Tableau, Power BI, or custom-built dashboards to provide real-time insights and support decision-making.

# # Level 4: Data Visualization and Reporting

# ## Task 4.1: Visualisation

# #### 1. Bar Chart: Number of Trains per Source Station

# In[37]:


# Count the number of trains per source station
source_station_counts = data['Source_Station_Name'].value_counts()

# Plot the bar chart
plt.figure(figsize=(12, 6))
source_station_counts.head(10).plot(kind='bar', color='blue')
plt.title('Number of Trains per Source Station (Top 10)')
plt.xlabel('Source Station')
plt.ylabel('Number of Trains')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# #### 2. Line Chart: Day-wise Distribution of Trains

# In[38]:


# Count the number of trains for each day of the week
trains_per_day = data['days'].value_counts().sort_index()

# Plot the line chart
plt.figure(figsize=(10, 6))
trains_per_day.plot(kind='line', marker='o', color='purple')
plt.title('Day-wise Distribution of Trains')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Trains')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# #### 3. Heatmap: Number of Trains between Source and Destination Stations

# In[41]:


import seaborn as sns

# Create a pivot table for heatmap
pivot_table = data.pivot_table(index='Source_Station_Name', columns='Destination_Station_Name', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=False, fmt='d')
plt.title('Number of Trains between Source and Destination Stations')
plt.xlabel('Destination Station')
plt.ylabel('Source Station')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# #### 4. Additional Bar Chart: Number of Trains per Destination Station

# In[42]:


# Count the number of trains per destination station
destination_station_counts = data['Destination_Station_Name'].value_counts()

# Plot the bar chart
plt.figure(figsize=(12, 6))
destination_station_counts.head(10).plot(kind='bar', color='salmon')
plt.title('Number of Trains per Destination Station (Top 10)')
plt.xlabel('Destination Station')
plt.ylabel('Number of Trains')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ## Task 4.2: Reporting

# # Comprehensive Report: Train Dataset Analysis
# 
# ## Introduction
# 
# This report presents an analysis of the train dataset, focusing on the distribution of train journeys across various stations and days of the week. The insights derived from this analysis aim to inform decision-making for optimizing train schedules, improving services at major hubs, and balancing passenger loads throughout the week.
# 
# ## Data Overview
# 
# ### The dataset contains the following columns:
# 
# Train_No
# 
# Train_Name
# 
# Source_Station_Name
# 
# Destination_Station_Name
# 
# days
# 
# ## Data Preprocessing
# 
# Loading the Dataset: The dataset was loaded using the pandas library.
# 
# Initial Exploration: Displayed the first few rows to understand the data structure.
# 
# Basic Structure and Missing Values
# 
# Data Types: The columns contain a mix of integers and strings.
# 
# Missing Values: No missing values were detected in the dataset.
# 
# ### Key Statistics
# 
# Total Number of Trains: 2674
# 
# Unique Source Stations: 122
# 
# Unique Destination Stations: 120
# 
# ### Visualizations and Insights
# 
# 1. Number of Trains per Source Station (Top 10)
# 
# Insight: Major hubs like specific source stations have the highest number of train departures, indicating significant passenger traffic.
# 
# 2. Day-wise Distribution of Trains
# 
# Insight: Certain days of the week have higher train frequencies, suggesting peak travel periods.
# 
# 3. Heatmap: Number of Trains between Source and Destination Stations
# 
# Insight: The heatmap highlights high-traffic routes and connections between specific source and destination stations.
# 
# 4. Number of Trains per Destination Station (Top 10)
# 
# ### Insight: Popular destinations like specific destination stations see a high volume of train arrivals, indicating significant passenger interest.
# 
# ### Correlation Analysis
# 
# To investigate the correlation between the number of trains and specific days of the week, we assigned numerical values to the days (e.g., Monday = 0, Tuesday = 1, etc.). The correlation coefficient was calculated, indicating the strength and direction of the relationship between days of the week and the number of train journeys.
# 
# ### Insight: The correlation coefficient indicates the strength and direction of the relationship between days of the week and the number of train journeys.
# 
# ## Recommendations
# 
# Optimize Train Schedules:
# 
# High Demand Days: Increase the number of trains on days with higher demand to accommodate passenger flow.
# 
# Low Demand Days: Reduce the number of trains or deploy smaller trains on days with lower demand to optimize

# In[ ]:




