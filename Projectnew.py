#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Downlaod relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
get_ipython().system('pip install ruptures')
import ruptures as rpt


# In[8]:


#Downlaod data set
data = pd.read_csv(r"C:\Users\uresha\Dropbox\PC\Desktop\UH\Project\Data\Death and Incidence.csv")

#Check first few raws
print(data.head(10))
print(data.tail(10))


# In[9]:


#PREPRCOSESSING

#Shape of the dataset
print(f'Death : {data.shape}')


# In[10]:


#Check types of the data and values
print(data.info())


# In[11]:


#Missing values
print(data.isnull().sum())


# In[12]:


#Summary statistics
print(data.describe())


# In[13]:


#Identify the duplicates
duplicates = data.duplicated().sum()
print(f'Number of duplicate rows: {duplicates}')


# In[14]:


#Rename the location name and cause name
data['location_name'] = data['location_name'].replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')

#Print the list of location to verify
print(data['location_name'].unique())


# In[15]:


# Create a mapping dictionary for cancer cause names and their abbreviations
can_cause_abbreviations = {
    "Bladder cancer": "BC",
    "Brain and central nervous system cancer": "BCNS",
    "Breast cancer": "BrC",
    "Cervical cancer": "CC",
    "Colon and rectum cancer": "CRC",
    "Esophageal cancer": "EsC",
    "Eye cancer": "EC",
    "Gallbladder and biliary tract cancer": "GBC",
    "Hodgkin lymphoma": "HL",
    "Kidney cancer": "KC",
    "Larynx cancer": "LC",
    "Leukemia": "Leuk",
    "Lip and oral cavity cancer": "LOCC",
    "Liver cancer": "LvrC",
    "Malignant neoplasm of bone and articular cartilage": "MNBAC",
    "Malignant skin melanoma": "MSM",
    "Mesothelioma": "Mes",
    "Multiple myeloma": "MM",
    "Nasopharynx cancer": "NC",
    "Neuroblastoma and other peripheral nervous cell tumors": "NPNCT",
    "Non-Hodgkin lymphoma": "NL",
    "Non-melanoma skin cancer": "NSC",
    "Other malignant neoplasms": "OMN",
    "Other neoplasms": "ON",
    "Other pharynx cancer": "OPC",
    "Ovarian cancer": "OC",
    "Pancreatic cancer": "PanC",
    "Prostate cancer": "ProsC",
    "Soft tissue and other extraosseous sarcomas": "STES",
    "Stomach cancer": "SC",
    "Testicular cancer": "TC",
    "Thyroid cancer": "ThyC",
    "Tracheal, bronchus, and lung cancer": "TBLC",
    "Uterine cancer": "UC"
}

#Create a new column for the abbreviations
data['cause_abbreviation'] = data['cause_name'].replace(can_cause_abbreviations)

#Check the updated DataFrame
print(data[['cause_name', 'cause_abbreviation']].head())


# In[16]:


#EDA

#COUNTRY WISE TOTAL COUNT 

#Number of entries for each country
Country_count = data['location_name'].value_counts()
print(Country_count)


#AGE WISE TOTAL COUNT 

#Number of entries for each age group
Age_count = data['age_name'].value_counts().sort_index()
print(Age_count)


# In[17]:


#DISTRIBUTION OF AGE GROUP AND GENDER BY COUNTRY

#Convert 'age_name' to categorical and sort to order
data['age_name'] = pd.Categorical(data['age_name'], categories = sorted(data['age_name'].unique()), ordered = True)

#Retrieve the top 12 countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Deaths' and 'Incidence'
palette_deaths = sns.color_palette("Set1", n_colors=2)  
palette_incidence = sns.color_palette("Set1", n_colors=2) 

#Separate graphs for each country and measure
for country in countries:
    #Create a new graph for each country
    plt.figure(figsize = (10, 10))  

    #Initialize a counter for subplots
    subplot_index = 1

    #Filter data for the current country
    country_data = data[data['location_name'] == country]

    #Separate 'Deaths' and 'Incidence'
    for measure in ['Deaths', 'Incidence']:
        measure_data = country_data[country_data['measure_name'] == measure]

        # Choose the palette based on the measure
        if measure == 'Deaths':
            palette = palette_deaths
        else:
            palette = palette_incidence

        # Create a subplot 
        plt.subplot(2, 1, subplot_index)

        # Create line plot for each measure with the chosen palette
        sns.lineplot(x = 'age_name', y = 'val', hue = 'sex_name', data=measure_data, palette = palette, marker = 'o', errorbar=('ci', 50))

        # Add labels, titles, and font sizes
        plt.title(f'{country} - {measure.capitalize()} by Age and Gender', fontsize = 14)
        plt.xlabel('Age Group', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)
        plt.xticks(rotation = 45, fontsize = 10)

        subplot_index += 1  

    #Print the plot
    plt.tight_layout()
    plt.show()


# In[18]:


#DISTRIBUTION OF GENDER BY COUNTRY

#Retrieve the countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Male' and 'Female'
palette_deaths = sns.color_palette("dark", n_colors=2)  
palette_incidence = sns.color_palette("pastel", n_colors=2)  

#Separate graphs for each country and measure
for country in countries:
    # Create a new graph for each country
    plt.figure(figsize=(8, 10)) 

    #Initialize a counter for subplots
    subplot_index = 1

    #Filter data for the current country
    country_data = data[data['location_name'] == country]

    #Separate 'Deaths' and 'Incidence'
    for measure in ['Deaths', 'Incidence']:
        measure_data = country_data[country_data['measure_name'] == measure]

        #Choose the palette based on the measure
        palette = palette_deaths if measure == 'Deaths' else palette_incidence

        #Create a subplot 
        plt.subplot(2, 1, subplot_index)

        #Create a multiple bar plot with age and gender distribution, without legend
        barplot = sns.barplot(
            x='sex_name', y = 'val', hue = 'sex_name', data = measure_data, palette = palette, errorbar=None, dodge = False, width = 0.6)
        
        #Set labels and title with font sizes
        plt.title(f'{country} - {measure.capitalize()} by Age and Gender', fontsize=14)
        plt.xlabel('Gender', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)

        #Set x-axis labels to 'Male' and 'Female'
        barplot.set_xticks([0, 1])
        barplot.set_xticklabels(['Male', 'Female'], fontsize=10)

        subplot_index += 1  

    #Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# In[19]:


#CAUSES OF DEATHS BY COUNTRY

#Retrieve the 12 countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Deaths' and 'Incidence'
#Choose a palette that has a sufficient number of colors for all categories
palette_deaths = sns.color_palette("viridis", n_colors=len(data['cause_abbreviation'].unique())) 
palette_incidence = sns.color_palette("Set2", n_colors=len(data['cause_abbreviation'].unique())) 

#Separate graphs for each country and measure
for country in countries:
    #Filter data for the current country
    country_data = data[data['location_name'] == country]

    #Create a new figure for the country
    plt.figure(figsize=(10, 10))

    #Initialize a counter for subplots
    subplot_index = 1

    #Separate 'Deaths' and 'Incidence'
    for measure in ['Deaths', 'Incidence']:
        measure_data = country_data[country_data['measure_name'] == measure]

        #Choose the palette based on the measure
        if measure == 'Deaths':
            palette = palette_deaths
        else:
            palette = palette_incidence

        #Create a subplot for each measure
        ax = plt.subplot(2, 1, subplot_index)

        #Create bar plot 
        sns.barplot(x='cause_abbreviation', y='val', data=measure_data, palette=palette, hue='cause_abbreviation', errorbar = None)
        
        plt.title(f'{country} - {measure.capitalize()} Distribution of Cancer Causes', fontsize=14)
        plt.xlabel('Cause of Cancer', fontsize=10)
        plt.ylabel('Value', fontsize=10)  
        plt.xticks(rotation=45, fontsize=10)

        #Add the value on top of each bar
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f'{p.get_height():.0f}', ha='center', va='bottom', fontsize=10)

        subplot_index += 1

    #Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# In[20]:


#CAUSE OF CANSER BY COUNTRY AND GENDER
    
#Convert 'age_name' to categorical and sort to order
data['age_name'] = pd.Categorical(data['age_name'], categories = sorted(data['age_name'].unique()), ordered = True)

#Retrieve the top 12 countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Deaths' and 'Incidence'
palette_deaths = sns.color_palette("dark", n_colors=2)  
palette_incidence = sns.color_palette("viridis", n_colors=2) 

#Separate graphs for each country and measure
for country in countries:
    #Create a new graph for each country
    plt.figure(figsize = (10, 12))  

    #Initialize a counter for subplots
    subplot_index = 1

    #Filter data for the current country
    country_data = data[data['location_name'] == country]

    #Separate 'Deaths' and 'Incidence'
    for measure in ['Deaths', 'Incidence']:
        measure_data = country_data[country_data['measure_name'] == measure]

        # Choose the palette based on the measure
        if measure == 'Deaths':
            palette = palette_deaths
        else:
            palette = palette_incidence

        # Create a subplot 
        plt.subplot(2, 1, subplot_index)

        # Create line plot for each measure with the chosen palette
        sns.lineplot(x = 'cause_abbreviation', y = 'val', hue = 'sex_name', data=measure_data, palette = palette, marker = 'o', errorbar=('ci', 50))

        # Add labels, titles, and font sizes
        plt.title(f'{country} - {measure.capitalize()} Distribution of Cancer Causes', fontsize = 14)
        plt.xlabel('Age Group', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)
        plt.xticks(rotation = 45, fontsize = 10)

        subplot_index += 1  

    #Print the plot
    plt.tight_layout()
    plt.show()


# In[21]:


#AGE GROUP BY GENDER

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


# In[22]:


#AGE GROUP BY CANCER CAUSE

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['cause_abbreviation'])

#Print the table
print(age_sex_crosstab)


# In[23]:


#AGE GROUP BY MEASURE NAME

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['measure_name'])

#Print the table
print(age_sex_crosstab)


# In[24]:


#AGE GROUP BY COUNTRY

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['location_name'])

#Print the table
print(age_sex_crosstab)


# In[25]:


#AGE GROUP BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['age_name'])

#Print the table
print(age_sex_crosstab)


# In[26]:


#GENDER BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


# In[27]:


#GENDER BY COUNTRY

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['location_name'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


# In[28]:


#COUNTRY BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['location_name'])

#Print the table
print(age_sex_crosstab)


# In[29]:


#GENDER BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


# In[30]:


#TREND BY YEAR AND MEASURE TYPE - GLOBALLY

plt.figure(figsize = (14, 8))
sns.lineplot(x = 'year', y = 'val', hue = 'measure_name', data = data, errorbar=('ci', 95), marker='o')
plt.title('Trends in Cancer Incidence and Mortality Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation = 45)
plt.legend(title='Measure Type')
plt.show()


# In[31]:


#TREND BY YEAR AND MEASURE TYPE - GLOBALLY_check clear view

plt.figure(figsize = (14, 8))
sns.lineplot(x = 'year', y = 'val', hue = 'measure_name', data = data, errorbar=('ci', 95), marker='o')
plt.title('Trends in Cancer Incidence and Mortality Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.ylim(0,1000)
plt.yticks(np.arange(0, 1001, 50))
plt.xticks(rotation = 45)
plt.legend(title='Measure Type')
plt.show()


# In[32]:


#MOTALITY RATE COMPARISON OVER TIME

#Dictionary to map each country to a specific marker
markers = {
    'Mexico': 'o',            
    'France': 's',            
    'Japan': 'D',              
    'Poland': '^',             
    'South Africa': 'x',       
    'Nigeria': 'p',            
    'Brazil': 'h',             
    'India': '+',              
    'United Kingdom': '>',      
    'China': '*',              
    'United States of America': '.',  
    'Australia': ',',         
}

plt.figure(figsize=(30, 30))

#Line plot with specific marker
for country, marker in markers.items():
    #Filter data for the country
    country_data = data[(data['location_name'] == country) & (data['measure_name'] == 'Deaths')]
    
    #Plot the line with confidence interval shading
    sns.lineplot(
        x='year', 
        y='val', 
        data=country_data, 
        marker=marker, 
        label=country, 
        dashes=False,
        alpha=0.5,
        markersize=10,  
        linewidth=3,
        errorbar=None
    )

    #Display the country name near the line
    y_pos = country_data['val'].mean()
    x_pos = country_data['year'].mean()
    plt.text(
        x_pos, 
        y_pos, 
        country, 
        color='black', 
        fontsize=15, 
        ha='center',  # Horizontal alignment of the label
        va='center',  # Vertical alignment of the label
        fontweight='bold'
    )

plt.title('Yearly Mortality Rate Comparison Across Countries', fontsize=24)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Mortality Rate', fontsize=20)
plt.ylim(0, 3001)
plt.yticks(range(0, 3000, 100))
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)


plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, title_fontsize=18)
plt.show()


# In[33]:


#MOTALITY RATE COMPARISON OVER TIME _ TO TAKE CLEAR VIEW CHINA REMOVED

#Dictionary to map each country to a specific marker
markers = {
    'Mexico': 'o',            
    'France': 's',            
    'Japan': 'D',              
    'Poland': '^',             
    'South Africa': 'x',       
    'Nigeria': 'p',            
    'Brazil': 'h',             
    'India': '+',              
    'United Kingdom': '>',      
    #'China': '*',              
    'United States of America': '.',  
    'Australia': ',',         
}

plt.figure(figsize=(30, 30))

#Line plot with specific marker
for country, marker in markers.items():
    #Filter data for the country
    country_data = data[(data['location_name'] == country) & (data['measure_name'] == 'Deaths')]
    
    #Plot the line with confidence interval shading
    sns.lineplot(
        x='year', 
        y='val', 
        data=country_data, 
        marker=marker, 
        label=country, 
        dashes=False,
        alpha=0.5,
        markersize=10,  
        linewidth=3,
        errorbar=None
    )

    #Display the country name near the line
    y_pos = country_data['val'].mean()
    x_pos = country_data['year'].mean()
    plt.text(
        x_pos, 
        y_pos, 
        country, 
        color='black', 
        fontsize=15, 
        ha='center',  
        va='center',  
        fontweight='bold'
    )

plt.title('Yearly Mortality Rate Comparison Across Countries', fontsize=24)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Mortality Rate', fontsize=20)
plt.ylim(0, 700)
plt.yticks(range(0, 700, 25))
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)


plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, title_fontsize=18)
plt.show()


# In[34]:


#YEARLY TREND FOR EACH CANCER CAUSE

plt.figure(figsize=(20, 16))
sns.lineplot(data=data, x='year', y='val', hue='cause_abbreviation', errorbar=None)
plt.title('Yearly Trends in Cancer Types')
plt.xlabel('Year')
plt.ylabel('Count')
plt.ylim(-500, 40000)
plt.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()


# In[35]:


#YEARLY TREND FOR EACH CANCER CAUSE

plt.figure(figsize=(20, 16))
sns.lineplot(data=data, x='year', y='val', hue='cause_abbreviation', errorbar=None)
plt.title('Yearly Trends in Cancer Types')
plt.xlabel('Year')
plt.ylabel('Count')
plt.ylim(0, 1000)
plt.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()


# In[36]:


#EXPLORING DIFFERENCES BY AGE AND YEAR

#Plot cancer cases by age group over the years
plt.figure(figsize=(14, 8))
sns.lineplot(data=data, x='year', y='val', hue='age_name', errorbar=None)
plt.title('Cancer Cases by Age Group Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[37]:


#CORRELATION BETWEEN AGE AND CANCER INCIDENCE #####################

#Filter data for incidence only
incidence_data = data[data['measure_name'] == 'Incidence']

#Creating pivot table for incidence data
correlation_data_incidence = incidence_data.pivot_table(values='val', index='age_name', columns='year', aggfunc='sum', observed='True')

#Calculating correlation matrix for incidence data
correlation_matrix_incidence = correlation_data_incidence.corr()

#Plot the heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_incidence, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'}, annot_kws={"fontsize": 8})
plt.title('Correlation Between Cancer Incidence Across Years')
plt.show()


# In[38]:


#MAP THE INCOME LEVEL TO COUNTRY

income_levels = {
    'South Africa': 'Upper Middle-Income',
    'Nigeria': 'Lower Middle-Income',
    'China': 'Upper Middle-Income',
    'India': 'Lower Middle-Income',
    'Japan': 'High-Income',
    'France': 'High-Income',
    'Poland': 'Upper Middle-Income',
    'United Kingdom': 'High-Income',
    'United States of America': 'High-Income',
    'Mexico': 'Upper Middle-Income',
    'Brazil': 'Upper Middl-Income',
    'Australia': 'High-Income'
}

#Create a new column 
data['income_level'] = data['location_name'].map(income_levels)

#Print data
print(data)


# In[39]:


#CANCER MORTALITY TREND BY INCOME LEVEL AND YEAR 

filtered_data = data[data['measure_name'].isin(['Deaths'])]

# Plot the trend by income level and year for deaths and incidence
plt.figure(figsize=(14, 8))
sns.lineplot(data=filtered_data, x='year', y='val', hue='income_level')
plt.title('Cancer Mortality Trends by Income Level')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Income Level')
plt.xticks(rotation=45)
plt.show()


# In[40]:


#CANCER INCIDENCE TREND BY INCOME LEVEL AND YEAR_Take clear view

filtered_data = data[data['measure_name'].isin(['Incidence'])]

# Plot the trend by income level and year for deaths and incidence
plt.figure(figsize=(14, 8))
sns.lineplot(data=filtered_data, x='year', y='val', hue='income_level')
plt.title('Cancer Incidence Trends by Income Level')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Income Level')
plt.xticks(rotation=45)
plt.show()


# In[41]:


#IDENTIFY WHETHER THERE IS ANY RELATIONSHIP BETWEEN DEMOGRAPIC FACTORS

#FIND THE CANCER MORTALITY - RELATIONSHIP BETWEEN GENDER AND LOCATION

#Filter data for "Deaths"
deaths_data = data[data['measure_name'] == 'Deaths']

#Creating a contingency table for Deaths
contingency_table_deaths = pd.crosstab(deaths_data['sex_name'], deaths_data['location_name'])
print("\nContingency Table for Deaths:")
print(contingency_table_deaths)

#Chi-square test 
chi2_deaths, p_deaths, dof_deaths, expected_deaths = stats.chi2_contingency(contingency_table_deaths)

# Print results for Deaths
print("\nResults for Deaths - - Gender & Country")
print(f"Chi-square Statistic: {chi2_deaths}")
print(f"P-value: {p_deaths}")
print(f"Degrees of Freedom: {dof_deaths}")
print("Expected Frequencies:\n", expected_deaths)


# In[42]:


#FIND THE CANCER INCIDENCE - RELATIONSHIP BETWEEN GENDER AND LOCATION

#Filter data for "Incidence"
incidence_data = data[data['measure_name'] == 'Incidence']

# Create a contingency table for Incidence
contingency_table_incidence = pd.crosstab(incidence_data['sex_name'], incidence_data['location_name'])
print("\nContingency Table for Incidence:")
print(contingency_table_incidence)

#Chi-square test for Incidence
chi2_incidence, p_incidence, dof_incidence, expected_incidence = stats.chi2_contingency(contingency_table_incidence)

# Print results
print("\nResults for Incidence - Gender & Country")
print(f"Chi-square Statistic: {chi2_incidence}")
print(f"P-value: {p_incidence}")
print(f"Degrees of Freedom: {dof_incidence}")
print("Expected Frequencies:\n", expected_incidence)


# In[43]:


#FIND THE CANCER MORTALITY - RELATIONSHIP BETWEEN GENDER AND AGE

#Filter data for "Deaths"
deaths_data = data[data['measure_name'] == 'Deaths']

#Creating a contingency table for Deaths
contingency_table_deaths = pd.crosstab(deaths_data['sex_name'], deaths_data['age_name'])
print("\nContingency Table for Deaths:")
print(contingency_table_deaths)

#Chi-square test 
chi2_deaths, p_deaths, dof_deaths, expected_deaths = stats.chi2_contingency(contingency_table_deaths)

#Print results
print("\nResults for Deaths - Gender & Age")
print(f"Chi-square Statistic: {chi2_deaths}")
print(f"P-value: {p_deaths}")
print(f"Degrees of Freedom: {dof_deaths}")
print("Expected Frequencies:\n", expected_deaths)


# In[44]:


#FIND THE CANCER INCIDENCE - RELATIONSHIP BETWEEN GENDER AND AGE

#Filter data for "Incidence"
incidence_data = data[data['measure_name'] == 'Incidence']

#Create a contingency table for Incidence
contingency_table_incidence = pd.crosstab(incidence_data['sex_name'], incidence_data['age_name'])
print("\nContingency Table for Incidence:")
print(contingency_table_incidence)

# Chi-square test for Incidence
chi2_incidence, p_incidence, dof_incidence, expected_incidence = stats.chi2_contingency(contingency_table_incidence)

#Print results
print("\nResults for Incidence - Gender & Age")
print(f"Chi-square Statistic: {chi2_incidence}")
print(f"P-value: {p_incidence}")
print(f"Degrees of Freedom: {dof_incidence}")
print("Expected Frequencies:\n", expected_incidence)


# In[45]:


#FIND THE CANCER MORTALITY - RELATIONSHIP BETWEEN GENDER AND INCOME LEVEL OF COUNTRY

#Filter data for "Deaths"
deaths_data = data[data['measure_name'] == 'Deaths']

#Creating a contingency table for Deaths
contingency_table_deaths = pd.crosstab(deaths_data['sex_name'], deaths_data['income_level'])
print("\nContingency Table for Deaths:")
print(contingency_table_deaths)

#Chi-square test 
chi2_deaths, p_deaths, dof_deaths, expected_deaths = stats.chi2_contingency(contingency_table_deaths)

#Print results
print("\nResults for Deaths - Gender & Income Level")
print(f"Chi-square Statistic: {chi2_deaths}")
print(f"P-value: {p_deaths}")
print(f"Degrees of Freedom: {dof_deaths}")
print("Expected Frequencies:\n", expected_deaths)


# In[46]:


#FIND THE CANCER INCIDENCE - RELATIONSHIP BETWEEN GENDER AND INCOME LEVEL OF COUNTRY

#Filter data for "Incidence"
incidence_data = data[data['measure_name'] == 'Incidence']

#Create a contingency table for Incidence
contingency_table_incidence = pd.crosstab(incidence_data['sex_name'], incidence_data['income_level'])
print("\nContingency Table for Incidence:")
print(contingency_table_incidence)

# Chi-square test for Incidence
chi2_incidence, p_incidence, dof_incidence, expected_incidence = stats.chi2_contingency(contingency_table_incidence)

#Print results 
print("\nResults for Incidence - Gender & Income Level")
print(f"Chi-square Statistic: {chi2_incidence}")
print(f"P-value: {p_incidence}")
print(f"Degrees of Freedom: {dof_incidence}")
print("Expected Frequencies:\n", expected_incidence)


# In[47]:


#FIND THE CANCER MORTALITY - RELATIONSHIP BETWEEN AGE AND INCOME LEVEL OF COUNTRY

#Filter data for "Deaths"
deaths_data = data[data['measure_name'] == 'Deaths']

#Creating a contingency table for Deaths
contingency_table_deaths = pd.crosstab(deaths_data['age_name'], deaths_data['income_level'])
print("\nContingency Table for Deaths:")
print(contingency_table_deaths)

#Chi-square test 
chi2_deaths, p_deaths, dof_deaths, expected_deaths = stats.chi2_contingency(contingency_table_deaths)

#Print results
print("\nResults for Deaths - Age & Income Level")
print(f"Chi-square Statistic: {chi2_deaths}")
print(f"P-value: {p_deaths}")
print(f"Degrees of Freedom: {dof_deaths}")
print("Expected Frequencies:\n", expected_deaths)


# In[48]:


#FIND THE CANCER INCIDENCE - RELATIONSHIP BETWEEN AGE AND INCOME LEVEL OF COUNTRY

#Filter data for "Incidence"
incidence_data = data[data['measure_name'] == 'Incidence']

# Create a contingency table for Incidence
contingency_table_incidence = pd.crosstab(incidence_data['age_name'], incidence_data['income_level'])
print("\nContingency Table for Incidence:")
print(contingency_table_incidence)

#Chi-square test
chi2_incidence, p_incidence, dof_incidence, expected_incidence = stats.chi2_contingency(contingency_table_incidence)

#Print results 
print("\nResults for Incidence - Age & Income Level")
print(f"Chi-square Statistic: {chi2_incidence}")
print(f"P-value: {p_incidence}")
print(f"Degrees of Freedom: {dof_incidence}")
print("Expected Frequencies:\n", expected_incidence)


# In[49]:


#FIND THE CANCER MORTALITY - RELATIONSHIP BETWEEN AGE AND COUNTRY

#Filter data for "Deaths"
deaths_data = data[data['measure_name'] == 'Deaths']

#Creating a contingency table for Deaths
contingency_table_deaths = pd.crosstab(deaths_data['age_name'], deaths_data['location_name'])
print("\nContingency Table for Deaths:")
print(contingency_table_deaths)

#Chi-square test 
chi2_deaths, p_deaths, dof_deaths, expected_deaths = stats.chi2_contingency(contingency_table_deaths)

#Print results
print("\nResults for Deaths - Age & Country")
print(f"Chi-square Statistic: {chi2_deaths}")
print(f"P-value: {p_deaths}")
print(f"Degrees of Freedom: {dof_deaths}")
print("Expected Frequencies:\n", expected_deaths)


# In[50]:


#FIND THE CANCER INCIDENCE - RELATIONSHIP BETWEEN AGE AND COUNTRY

#Filter data for "Incidence"
incidence_data = data[data['measure_name'] == 'Incidence']

#Create a contingency table 
contingency_table_incidence = pd.crosstab(incidence_data['age_name'], incidence_data['location_name'])
print("\nContingency Table for Incidence:")
print(contingency_table_incidence)

#Chi-square test
chi2_incidence, p_incidence, dof_incidence, expected_incidence = stats.chi2_contingency(contingency_table_incidence)

#Print results 
print("\nResults for Incidence - Age & Country")
print(f"Chi-square Statistic: {chi2_incidence}")
print(f"P-value: {p_incidence}")
print(f"Degrees of Freedom: {dof_incidence}")
print("Expected Frequencies:\n", expected_incidence)


# In[51]:


#EFFECT ON INCOME LEVEL ON CANCER COUNT 

#Function to perform ANOVA 
def perform_anova(data, measure_name):
    # Filter data for the specific measure
    groups = data[data['measure_name'] == measure_name].groupby('income_level')['val'].apply(list)
    
    #F value, P value
    f_statistic, p_value = stats.f_oneway(*groups)

    #Print results
    print(f"Results for {measure_name}:")
    print(f"F-Statistic: {f_statistic:.4f}, P-value: {p_value:.4f}")
    print("Reject the null hypothesis." if p_value < 0.05 else "Accept the null hypothesis.")
    print("=" * 50)

#ANOVA for both Deaths and Incidence
for measure in ["Deaths", "Incidence"]:
    perform_anova(data, measure) 


# In[52]:


#POST-HOC ANALYSIS

#Recoding dictionary
income_id = {
    'Upper Middle-Income': 1,
    'Lower Middle-Income': 2,
    'High-Income': 3
}

#Recode the income_level in new column
data['income_id'] = data['income_level'].replace(income_id)

#Display the updated DataFrame
print(data)


# In[53]:


#Tukey's HSD post-hoc test

def perform_posthoc(data, measure_name):
    measure_data = data[data['measure_name'] == measure_name].copy()  # Create a copy to avoid modifying the original

    #Convert 'income_id' to numeric
    measure_data.loc[:, 'income_id'] = pd.to_numeric(measure_data['income_id'], errors='coerce')

    #Optionally drop rows where 'income_id' is NaN (if any)
    measure_data = measure_data.dropna(subset=['income_id'])

    #Check 'income_id' is categorical after conversion
    measure_data.loc[:, 'income_id'] = measure_data['income_id'].astype('category')

    #Fit the model
    model = ols('val ~ C(income_id)', data=measure_data).fit()

    #Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(endog=measure_data['val'], groups=measure_data['income_id'], alpha=0.05)

    #Print Tukey's HSD results
    print(f"\nTukey's HSD Results for {measure_name}:")
    print(tukey_results)
    print("=" * 50)

#Perform Tukey's HSD for "Deaths" and "Incidence"
for measure in ["Deaths", "Incidence"]:
    perform_posthoc(data, measure)


# In[54]:


#EFFECT ON AGE ON CANCER COUNT 

#Function to perform ANOVA 
def perform_anova(data, measure_name):
    # Filter data for the specific measure
    groups = data[data['measure_name'] == measure_name].groupby('age_name', observed=True)['val'].apply(list)
    
    #F value, P value
    f_statistic, p_value = stats.f_oneway(*groups)

    #Print results
    print(f"Results for {measure_name}:")
    print(f"F-Statistic: {f_statistic:.4f}, P-value: {p_value:.4f}")
    print("Reject the null hypothesis." if p_value < 0.05 else "Accept the null hypothesis.")
    print("=" * 50)

#ANOVA for both Deaths and Incidence
for measure in ["Deaths", "Incidence"]:
    perform_anova(data, measure)


# In[55]:


#POST-HOC ANALYSIS

#Tukey's HSD post-hoc test
def perform_posthoc(data, measure_name):
    
    measure_data = data[data['measure_name'] == measure_name]

    #Fit the model
    model = ols('val ~ C(age_name)', data=measure_data).fit()

    #Perform Tukey's HSD
    tukey_results = pairwise_tukeyhsd(endog=measure_data['val'], groups=measure_data['age_name'], alpha=0.05)

    #Print Tukey's HSD results
    print(f"\nTukey's HSD Results for {measure_name}:")
    print(tukey_results)
    print("=" * 50)

# Perform Tukey's HSD for both Deaths and Incidence
for measure in ["Deaths", "Incidence"]:
    perform_posthoc(data, measure)


# In[56]:


#DICKEY-FULLER TEST

#Reduce dataset size for the test
#subset = data['val'].iloc[:10000] 

#Perform ADF test
#adf_result = adfuller(subset)

#Extract and interpret results
#print("ADF Statistic:", adf_result[0])
#print("p-value:", adf_result[1])
#print("Critical Values:", adf_result[4])

#if adf_result[1] < 0.05:
 #   print("The time series is non stationary (reject null hypothesis).")
#else:
  #  print("The time series is stationary (fail to reject null hypothesis).")


# In[58]:


#DICKEY FULLER TEST - DEATHS

#Filter the data for 'Deaths'
deaths_data = data[data['measure_name'] == 'Deaths']

#Aggregate the data by year
aggregated_deaths = deaths_data.groupby('year')['val'].sum()

#Perform the ADF test 
adf_result = adfuller(aggregated_deaths)

#Print the results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])

if adf_result[1] < 0.05:
    print("The time series is not stationary - data has unit root")
else:
    print("The time series is stationary - data has no unit root")


# In[116]:


#DICKEY FULLER TEST - INCIDENCE

#Filter the data for 'Deaths'
deaths_incidence = data[data['measure_name'] == 'Incidence']

#Aggregate the data by year
aggregated_incidence = deaths_data.groupby('year')['val'].sum()

#Perform the ADF test 
adf_result = adfuller(aggregated_deaths)

#Print the results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])

if adf_result[1] < 0.05:
    print("The time series is not stationary - data has unit root")
else:
    print("The time series is stationary - data has no unit root")


# In[159]:


#TIME SERIES ANALYSIS

#ARIMA MODEL - DEATHS

#Aggregate Deaths by year
deaths_series = data[data['measure_name'] == 'Deaths'].groupby('year')['val'].sum().reset_index()

#Prepare time series for Deaths
deaths_series.set_index('year', inplace=True)
deaths_series.index = pd.to_datetime(deaths_series.index, format='%Y')

#Define function
def run_arima_and_plot(series, title, color_historical, color_forecast):
    #Fit ARIMA model
    model = ARIMA(series['val'], order=(1, 1, 1))  # Adjust the (p, d, q) order as needed
    result = model.fit()
    
    #Forecast the next 10 years
    forecast = result.forecast(steps=10)
    
    #Print results
    print(f"\nARIMA Model Results for {title}:")
    print(result.summary())
    print(f"Forecasted Values ({title}):")
    print(forecast)
    
    #Plot historical and forecasted values
    historical_years = series.index
    forecast_years = pd.date_range(start=historical_years[-1] + pd.offsets.YearEnd(1), periods=10, freq='YE')
    historical_values = series['val']
    
    plt.figure(figsize=(12, 6))
    plt.plot(historical_years, historical_values, label=f"Historical {title}", color=color_historical)
    plt.plot(forecast_years, forecast, label=f"Forecasted {title}", color=color_forecast, linestyle='--')
    plt.title(f"Time Series Analysis Forecast: Deaths ARIMA Model")
    plt.xlabel("Year")
    plt.ylabel(title)
    plt.legend()
    plt.grid()
    plt.show()

# Run for Deaths
run_arima_and_plot(deaths_series, "Deaths", color_historical='blue', color_forecast='red')


# In[203]:


# Aggregate Incidence by year
incidence_series = data[data['measure_name'] == 'Incidence'].groupby('year')['val'].sum().reset_index()

# Prepare time series for Incidence
incidence_series.set_index('year', inplace=True)
incidence_series.index = pd.to_datetime(incidence_series.index, format='%Y')

# Define function
def run_arima_and_plot(series, title, color_historical, color_forecast):
    # Fit ARIMA model
    model = ARIMA(series['val'], order=(1, 1, 1))  
    result = model.fit()
    
    # Forecast the next 10 years
    forecast = result.forecast(steps=10)
    
    # Print results
    print(f"\nARIMA Model Results for {title}:")
    print(result.summary())
    print(f"Forecasted Values ({title}):")
    print(forecast)
    
    # Plot historical and forecasted values
    historical_years = series.index
    forecast_years = pd.date_range(start=historical_years[-1] + pd.offsets.YearEnd(0), periods=10, freq='YE')
    historical_values = series['val']
    
    plt.figure(figsize=(12, 6))
    plt.plot(historical_years, historical_values, label=f"Historical {title}", color=color_historical)
    plt.plot(forecast_years, forecast, label=f"Forecasted {title}", color=color_forecast, linestyle='--')
    plt.title(f"Time Series Analysis Forecast: {title} ARIMA Model")
    plt.xlabel("Year")
    #plt.xticks(range(2000, 2033, 2))
    plt.ylabel(title)
    plt.legend()
    plt.grid()
    plt.show()

# Run for Incidence
run_arima_and_plot(incidence_series, "Incidence", color_historical='green', color_forecast='orange')


# In[ ]:




