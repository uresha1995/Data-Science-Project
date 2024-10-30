#Downlaod relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

plt.figure(figsize=(14, 8))
sns.lineplot(data=data, x='year', y='val', hue='age_name', ci=None)
plt.title('Cancer Cases by Age Group Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
