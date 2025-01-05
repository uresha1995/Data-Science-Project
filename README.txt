
*Title of the Project

Analysing Global Health Trends: A Time Series Analysis of Cancer Incidence and Mortality Rate

*Overview 

This project involves a detailed analysis of global cancer incidence and mortality rates using a dataset containing over 500,000 entries. The analysis aims to identify trends and patterns in cancer incidence and mortality rate, with a specific focus on demographic factors like age, gender, and country with specific cancer types. And also dataset include the colomns with encoded data for each variable type.

Dataset
The dataset, titled "Death and Incidence.csv", contains the following columns:

1. location_name: Country or region where the data was recorded.
2. age_name: Age group of the population.
3. sex_name: Gender of the population.
4. cause_name: Type of cancer or cause of death.
5. measure_name: Measure type (either 'Deaths' or 'Incidence').
6. val: Value representing the count for deaths or incidences.
7. lower: Lower bound of the confidence interval.
8. upper: Upper bound of the confidence interval.

*Research Question

What are the trends in global cancer incidence and mortality rates over the past two decades, and how do these trends differ across various demographics, such as income levels, geographic regions, age and gender?

*Research Objectives

Analyze and identify trends in global cancer incidence and mortality rates (2000- 2021).
Identifying disparities based on demographic data.
Forecast future trends in cancer incidence and mortality using time series analysis.

*Libraries Used

Pandas: For data manipulation and analysis.
Matplotlib and Seaborn: For data visualization.
Scipy: For statistical tests and calculations.
Statsmodels: For regression analysis and hypothesis testing.

### Coding ##

*Data Cleaning and Preprocessing:

Renamed values for consistency (e.g., "United Kingdom of Great Britain and Northern Ireland" replaced with "United Kingdom").
REPLACED ---> United Kingdom of Great Britain and Northern Ireland => United Kingdom

Added a column for cancer type abbreviations for easier analysis.

ABBREVATION LIST FOR cause_name

    Bladder cancer = BC,
    Brain and central nervous system cancer = BCNS,
    Breast cancer = BrC,
    Cervical cancer = CC,
    Colon and rectum cancer = CRC,
    Esophageal cancer = EsC,
    Eye cancer = EC,
    Gallbladder and biliary tract cancer = GBC,
    Hodgkin lymphoma = HL,
    Kidney cancer = KC,
    Larynx cancer = LC,
    Leukemia = Leuk,
    Lip and oral cavity cancer = LOCC,
    Liver cancer = LvrC,
    Malignant neoplasm of bone and articular cartilage = MNBAC,
    Malignant skin melanoma = MSM,
    Mesothelioma = Mes,
    Multiple myeloma = MM,
    Nasopharynx cancer = NC,
    Neuroblastoma and other peripheral nervous cell tumors = NPNCT,
    Non-Hodgkin lymphoma = NL,
    Non-melanoma skin cancer = NSC,
    Other malignant neoplasms = OMN,
    Other neoplasms = ON,
    Other pharynx cancer = OPC,
    Ovarian cancer = OC,
    Pancreatic cancer = PanC,
    Prostate cancer = ProsC,
    Soft tissue and other extraosseous sarcomas = STES,
    Stomach cancer = SC,
    Testicular cancer = TC,
    Thyroid cancer = ThyC,
    Tracheal, bronchus, and lung cancer = TBLC,
    Uterine cancer = UC

*Income Level Mapping

Mapping countries to their income levels (e.g., Upper Middle-Income, High-Income) 

*Exploratory Data Analysis (EDA):

Checked for missing values, duplicates, and data types.
Performed summary statistics to understand the distribution and structure of the data.

*Generated Graphs using line graph, bar charts, heat maps:

Country-wise and Age-wise Analysis
Distribution of Cancer Types
Gender-based Analysis
Cause-specific Cancer Distribution
Trends in cancer incidence and mortality over time
Mortality rates across multiple countries over time
Yearly Trends for Each Cancer Cause
Cancer Cases by Age Group Over Time
Incidence and mortality by Income Level

*Cross-Tabulations:

Age group by gender
Age group by cancer cause
Age group by measure name (incidence or mortality)
Age group by country
Gender by cancer cause
Gender by country
Country by cancer cause
Gender and location (country)
Gender and age group
Gender and income level
Age group and income level
Age group and country

*Contingency Table

Contingency tables for demographic relationships in cancer incidence and mortality.

**ANOVA: A one-way Analysis of Variance (ANOVA) to test whether there is a significant difference in cancer counts (deaths and incidence) between different income levels and age groups.

**Post-Hoc Analysis: A Tukey’s HSD (Honest Significant Difference) test for pairwise comparisons following the ANOVA to further explore which specific income levels or age groups differ.

*Time Series Analysis for deaths 

ARIMA(1,1,1) - forecast the cancer incidence and mortality
Auto ARIMA - Identify (0,2,0) as best model still AIC is high
Did manual tuning to check whether other models have low AIC

Found best model is (0,2,0)

*Time Series Analysis for incidence

ARIMA(1,1,1) - forecast the cancer incidence and mortality
Auto ARIMA - Identify (0,1,0) as best model still AIC is high
Did manual tuning to check whether other models have low AIC

Found (0,2,0) is the best model

*Run test and training model by separate 80% training and 20% testing

Used to validate the model's performance by comparing predicted values to observed values

*Rolling Forecast

Run to validate the model's prediction ability
 
