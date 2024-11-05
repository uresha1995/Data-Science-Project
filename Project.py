#Downlaod relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


#Downlaod data set
data = pd.read_csv(r"C:\Users\uresha\Dropbox\PC\Desktop\UH\Project\Data\Death and Incidence.csv")

#Check first few raws
print(data.head(10))
print(data.tail(10))

#EDA

#Shape of the dataset
print(f'Death : {data.shape}')

#Check types of the data and values
print(data.info())

#Missing values
print(data.isnull().sum())

#Summary statistics
print(data.describe())

duplicates = data.duplicated().sum()
print(f'Number of duplicate rows: {duplicates}')

#Rename the location name and cause name
data['location_name'] = data['location_name'].replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')

#Print the list of location to verify
print(data['location_name'].unique())


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


#EDA

#COUNTRY WISE TOTAL COUNT 

#Number of entries for each country
Country_count = data['location_name'].value_counts()
print(Country_count)


#AGE WISE TOTAL COUNT 

#Number of entries for each age group
Age_count = data['age_name'].value_counts().sort_index()
print(Age_count)


#DISTRIBUTION OF AGE GROUP AND GENDER BY COUNTRY

#Convert 'age_name' to categorical and sort to order
data['age_name'] = pd.Categorical(data['age_name'], categories = sorted(data['age_name'].unique()), ordered = True)

#Retrieve the top 12 countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Deaths' and 'Incidence'
palette_deaths = sns.color_palette("viridis", n_colors=8)  
palette_incidence = sns.color_palette("magma", n_colors=8) 

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
        sns.lineplot(x = 'age_name', y = 'val', hue = 'sex_name', data=measure_data, palette = palette, marker = 'o', ci = 50)

        # Add labels, titles, and font sizes
        plt.title(f'{country} - {measure.capitalize()} by Age and Gender', fontsize = 14)
        plt.xlabel('Age Group', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)
        plt.xticks(rotation = 45, fontsize = 10)

        subplot_index += 1  

    #Print the plot
    plt.tight_layout()
    plt.show()
    
    
#DISTRIBUTION OF GENDER BY COUNTRY

#Retrieve the countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Male' and 'Female'
palette_deaths = sns.color_palette("dark", n_colors=8)  
palette_incidence = sns.color_palette("pastel", n_colors=8)  

#Separate graphs for each country and measure
for country in countries:
    # Create a new graph for each country
    plt.figure(figsize=(10, 12)) 

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
            x='sex_name', y = 'val', hue = 'sex_name', data = measure_data, palette = palette, ci = None, dodge = False, width = 0.6)
        
        #Set labels and title with font sizes
        plt.title(f'{country} - {measure.capitalize()} by Age and Gender', fontsize=14)
        plt.xlabel('Gender', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)

        #Set x-axis labels to 'Male' and 'Female'
        barplot.set_xticklabels(['Male', 'Female'], fontsize=10)

        subplot_index += 1  

    #Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


#CAUSES OF DEATHS BY COUNTRY
    
#Retrieve the 12 countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Deaths' and 'Incidence'
palette_deaths = sns.color_palette("viridis", n_colors=8)  
palette_incidence = sns.color_palette("Set2", n_colors=8)  

#Separate graphs for each country and measure
for country in countries:
    #Filter data for the current country
    country_data = data[data['location_name'] == country]

    #Create a new figure for the country
    plt.figure(figsize=(10, 12))

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
        plt.subplot(2, 1, subplot_index)

        #Create a bar plot for the distribution of cancer causes
        sns.countplot(x='cause_abbreviation', data=measure_data, palette=palette, width=0.5)

        #Add labels, titles, and font sizes
        plt.title(f'{country} - {measure.capitalize()} Distribution of Cancer Causes', fontsize=14)
        plt.xlabel('Cause of Cancer', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.xticks(rotation=45, fontsize=10)

        subplot_index += 1

    #Print the plot
    plt.tight_layout()
    plt.show()
    
#CAUSE OF CANSER BY COUNTRY AND GENDER
    
#Convert 'age_name' to categorical and sort to order
data['age_name'] = pd.Categorical(data['age_name'], categories = sorted(data['age_name'].unique()), ordered = True)

#Retrieve the top 12 countries
countries = data['location_name'].value_counts().index[:12]

#Define color palettes for 'Deaths' and 'Incidence'
palette_deaths = sns.color_palette("dark", n_colors=8)  
palette_incidence = sns.color_palette("viridis", n_colors=8) 

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
        sns.lineplot(x = 'cause_abbreviation', y = 'val', hue = 'sex_name', data=measure_data, palette = palette, marker = 'o', ci = 50)

        # Add labels, titles, and font sizes
        plt.title(f'{country} - {measure.capitalize()} Distribution of Cancer Causes', fontsize = 14)
        plt.xlabel('Age Group', fontsize = 10)
        plt.ylabel('Count', fontsize = 10)
        plt.xticks(rotation = 45, fontsize = 10)

        subplot_index += 1  

    #Print the plot
    plt.tight_layout()
    plt.show()


#AGE GROUP BY GENDER

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


#AGE GROUP BY CANCER CAUSE

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['cause_abbreviation'])

#Print the table
print(age_sex_crosstab)


#AGE GROUP BY MEASURE NAME

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['measure_name'])

#Print the table
print(age_sex_crosstab)


#AGE GROUP BY COUNTRY

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['age_name'], data['location_name'])

#Print the table
print(age_sex_crosstab)


#AGE GROUP BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['age_name'])

#Print the table
print(age_sex_crosstab)


#GENDER BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


#GENDER BY COUNTRY

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['location_name'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


#COUNTRY BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['location_name'])

#Print the table
print(age_sex_crosstab)


#GENDER BY CAUSES OF DEATH

#Create a cross-tabulation of age by sex
age_sex_crosstab = pd.crosstab(data['cause_abbreviation'], data['sex_name'])

#Print the table
print(age_sex_crosstab)


#TREND BY YEAR AND MEASURE TYPE

plt.figure(figsize = (14, 8))
sns.lineplot(x = 'year', y = 'val', hue = 'measure_name', data = data, ci = 95, marker='o')
plt.title('Trends in Cancer Incidence and Mortality Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation = 45)
plt.legend(title='Measure Type')
plt.show()

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

plt.figure(figsize=(25, 20))

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
        alpha=0.5          
    )

plt.title('Yearly Mortality Rate Comparison Across Countries', fontsize=24)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Mortality Rate', fontsize=20)
plt.ylim(0, 1000)


plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, title_fontsize=18)
plt.show()


#YEARLY TREND FOR EACH CANCER CAUSE

plt.figure(figsize=(20, 16))
sns.lineplot(data=data, x='year', y='val', hue='cause_name', ci=None)
plt.title('Yearly Trends in Cancer Types')
plt.xlabel('Year')
plt.ylabel('Count')
plt.ylim(-500, 40000)
plt.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()


#EXPLORING DIFFERENCES BY AGE AND YEAR

#Plot cancer cases by age group over the years
plt.figure(figsize=(14, 8))
sns.lineplot(data=data, x='year', y='val', hue='age_name', ci=None)
plt.title('Cancer Cases by Age Group Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# CORRELATION BETWEEN AGE AND CANCER MORTALITY

#Filter data for deaths only
deaths_data = data[data['measure_name'] == 'Deaths']

#Creating pivot table for deaths data
correlation_data_deaths = deaths_data.pivot_table(values='val', index='age_name', columns='year', aggfunc='sum')

#Calculating correlation matrix for deaths data
correlation_matrix_deaths = correlation_data_deaths.corr()

#Plot the heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_deaths, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'}, annot_kws={"fontsize": 8})
plt.title('Correlation Between Cancer Mortality Across Years')
plt.show()


# CORRELATION BETWEEN AGE AND CANCER INCIDENCE

#Filter data for incidence only
incidence_data = data[data['measure_name'] == 'Incidence']

#Creating pivot table for incidence data
correlation_data_incidence = incidence_data.pivot_table(values='val', index='age_name', columns='year', aggfunc='sum')

#Calculating correlation matrix for incidence data
correlation_matrix_incidence = correlation_data_incidence.corr()

#Plot the heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_incidence, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'}, annot_kws={"fontsize": 8})
plt.title('Correlation Between Cancer Incidence Across Years')
plt.show()

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
    'Brazil': 'Upper Middle-Income',
    'Australia': 'High-Income'
}

#Create a new column 
data['income_level'] = data['location_name'].map(income_levels)

#Print data
print(data)

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


#CANCER INCIDENCE TREND BY INCOME LEVEL AND YEAR 

filtered_data = data[data['measure_name'].isin(['Incidence'])]

# Plot the trend by income level and year for deaths and incidence
plt.figure(figsize=(14, 8))
sns.lineplot(data=filtered_data, x='year', y='val', hue='income_level')
plt.title('Cancer Incidnec Trends by Income Level')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Income Level')
plt.xticks(rotation=45)
plt.show()


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


#Tukey's HSD post-hoc test
def perform_posthoc(data, measure_name):
    
    measure_data = data[data['measure_name'] == measure_name]

    #Fit the model
    model = ols('val ~ C(income_id)', data=measure_data).fit()

    #Perform Tukey's HSD
    tukey_results = pairwise_tukeyhsd(endog=measure_data['val'], groups=measure_data['income_id'], alpha=0.05)

    #Print Tukey's HSD results
    print(f"\nTukey's HSD Results for {measure_name}:")
    print(tukey_results)
    print("=" * 50)

#Perform Tukey's HSD
for measure in ["Deaths", "Incidence"]:
    perform_posthoc(data, measure)
    

#EFFECT ON AGE ON CANCER COUNT 

#Function to perform ANOVA 
def perform_anova(data, measure_name):
    # Filter data for the specific measure
    groups = data[data['measure_name'] == measure_name].groupby('age_name')['val'].apply(list)
    
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
    
    
