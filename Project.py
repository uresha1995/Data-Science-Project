#Downlaod relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Downlaod data set
death = pd.read_csv("C:\\Users\\uresha\\Dropbox\\PC\\Desktop\\UH\\Project\\Death.csv")
incidence = pd.read_csv("C:\\Users\\uresha\\Dropbox\\PC\\Desktop\\UH\\Project\\Incidence.csv")

#Check first few raws
print(death.head())
print(incidence.head())

#EDA

#Shape of the dataset
print(f'Death : {death.shape}')
print(f'Incidence : {incidence.shape}')

#Check types of the data and values
print(death.info())
print(incidence.info())

#Missing values
print(death.isnull().sum())
print(incidence.isnull().sum())

#Summary statistics
print(death.describe())
print(incidence.describe())

# Unique countries in each dataset
print(f"Death : {death['location_name'].unique()}")
print(f"Incidence : {incidence['location_name'].unique()}")

#Map incidence country names to death country names
country_mapping = {
    "People's Republic of China": "China",
    'French Republic': 'France',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Republic of India': 'India',
    'Republic of South Africa': 'South Africa'}

#Replace location_name of the incidence dataset
incidence['location_name'] = incidence['location_name'].replace(country_mapping)

print(f"Death : {death['location_name'].unique()}")
print(f"Incidence : {incidence['location_name'].unique()}")


#Select variables  for dataset-new
death_new = death[['measure_name', 'location_name', 'sex_name', 'age_name', 'year', 'cause_name', 'val']] 
incidence_new = incidence[['measure_name', 'location_name', 'sex_name', 'age_name', 'year', 'cause_name', 'val']] 

#Use pd.concat to append the rows from death under incidence
merged = pd.concat([incidence_new, death_new], axis=0)

#Check the result
print(merged.head())
print(merged.tail(10))

# Relationship between cancer incidence and death rates
#Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='val', y='val', hue='location_name', data=merged) 
plt.title('Relationship Between Cancer Incidence and Mortality')
plt.xlabel('Cancer Incidence Rate')
plt.ylabel('Cancer Mortality Rate')
plt.show()





################
# Correlation matrix
death_values = death[['location_name', 'year','val']].copy()
death_values.rename(columns={'val': 'death_val'}, inplace=True)

incidence_values = incidence[['location_name', 'year','val']].copy()
incidence_values.rename(columns={'val': 'incidence_val'}, inplace=True)

merged_additional= pd.merge(death_values, incidence_values, on=['location_name', 'year'])

# Calculate the correlation matrix between death_val and incidence_val
corr_matrix = merged_additional[['death_val', 'incidence_val']].corr()
print(corr_matrix)

# Heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix: Relationship Between Cancer Incidence and Mortality', fontsize=16)
plt.show()

