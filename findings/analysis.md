# The project report and analysis : Detect faluty water pumps in Tanzania

# Table of contents

1. The objecttive of this project
2. Data details
3. Domain 
4. Exploring and cleaning data
5. Feature engneering
6. Modeling and evaluation
7. Challenges and augmentations



# 1.The objective of this project

Predict which pumps are functional, which need some repairs, and which don't work by using given dataset from the competition platform. [Driven Data website](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)


# 2. Data details


Data size : 59400

Predictor variables : 42

Labels : multi labels (functional, non functional, needs repair)

Created and recorded : between Oct 2002 and Dec 2012
![date_time](/image/date_time.PNG)

Feature varibale details: continuous, discrete and categorical varibles

* amount_tsh - Total static head (amount water available to waterpoint)
* date_recorded - The date the row was entered
* funder - Who funded the well
* gps_height - Altitude of the well
* installer - Organization that installed the well
* longitude - GPS coordinate
* latitude - GPS coordinate
* wpt_name - Name of the waterpoint if there is one
* num_private -
* basin - Geographic water basin
* subvillage - Geographic location
* region - Geographic location
* region_code - Geographic location (coded)
* district_code - Geographic location (coded)
* lga - Geographic location
* ward - Geographic location
* population - Population around the well
* public_meeting - True/False
* recorded_by - Group entering this row of data
* scheme_management - Who operates the waterpoint
* scheme_name - Who operates the waterpoint
* permit - If the waterpoint is permitted
* construction_year - Year the waterpoint was constructed
* extraction_type - The kind of extraction the waterpoint uses
* extraction_type_group - The kind of extraction the waterpoint uses
* extraction_type_class - The kind of extraction the waterpoint uses
* management - How the waterpoint is managed
* management_group - How the waterpoint is managed
* payment - What the water costs
* payment_type - What the water costs
* water_quality - The quality of the water
* quality_group - The quality of the water
* quantity - The quantity of water
* quantity_group - The quantity of water
* source - The source of the water
* source_type - The source of the water
* source_class - The source of the water
* waterpoint_type - The kind of waterpoint
* waterpoint_type_group - The kind of waterpoint

# 3. Domain information

## Causes of Well Problems

1. Degradiation: over 10 years? maybe time to change a part of pumps
2. Lack of calling water
3. Frozen water and lower the water level
4. Impurity (sands etc) causes pump problems
5. Most handpumps are manufactured in India with poor quality controls and recycled parts.
6. Inaccurate placement of groundwater extraction boreholes.
7. The shortage of skilled labor for placement and maintenance.

[reference](https://www1.agric.gov.ab.ca/$department/deptdocs.nsf/ba3468a2a8681f69872569d60073fde1/b235a3f65b62081b87256a5a005f5446/$FILE/WaterWells_module7.pdf)
[reference](http://www.rapidservicellc.com/manchester-plumbing/18-plumbing/1633-5-causes-of-well-pump-failure.html)


# 4. Exploring and cleaning data

1. Handling imbalanced data : Implimented resampling by SMOTE.
2. Missing values and replacement : Get median values out of each labels and placed with that values. 
3. Many of columns are duplicated. Droppped similar values. 



## Imbalanced dataset and resampling

![data_balance2](/image/data_balance2.PNG)
![data_balance](/image/data_balance.PNG)

After resampling, the dataset looks as below. Resampling techniques are applied to **only train data** after being splited.

![data_balance4](/image/data_balance4.PNG)
![data_balance3](/image/data_balance3.PNG)


## Are values missing at random or not? 

**Not at random**
* Population and contruction year 
* GPS height and population 
* installer and funder 

![missing_data](/image/missing_heatmap.PNG) 

![missing_data](/image/missing_dendrogram.PNG)

![missing_data](/image/missing_matrix.PNG)



## Popoulation Zero Region
* Pwani, Singida,Arusha (Zero)
* Kigoma, Rukwa (Nearly Zero)

![pop_zero](/image/population_zero.PNG)

## Population zero and construction year correlation 
* Label 0 : Non population zero
* Label 1 : Population zero 
* "Popularion is zero" could indicate that it can be a missing value and corelated to "construction year unknown" value. However, it is difficult to determine if this is a missing value or a case that no one lives in the certain areas. 

![pop_year](/image/population_year.PNG)


## Dry water pumps could cause "non functional "

![dry_water](/image/dry_water.PNG)

## Water quality

* Most of the water quality is soft (good)
* water_quality : Unknow and salty seems to be good predictors

![water_quality](/image/water_quality.PNG)


## Payment status

![payment_status](/image/payment_status.PNG)



## Funder and installer Top10

* DWE contributes to installing and funding the pumps the most. 

![installer](/image/installer.PNG)

![funder](/image/funder.PNG)

## Pump type: non func group has majority of "other"


![pump_type](/image/pump_type.PNG)


## The source of the water : "Unknown", "Salty water" and "Milky" in  Shallow well and Machine dbh

![pump_type](/image/water_source.PNG)

## Correlation : Labels and features  

**correlation with status_group_functional**
![corr_func](/image/corr_with_func.PNG)

**correlation wihth status_group_functional_needs_repair**

![corr_nonfunc](/image/corr_with_nonfunc.PNG)


**correlation wiht status_group_non_functional**

![corr_repair](/image/corr_with_repair.PNG)


## Folium interactive map
## See interactive map [https://nbviewer.jupyter.org/github/yukaberry/portfolio_tanzania_water_pumps/blob/master/folium_map_layercontrol.ipynb]
![foliummap](/image/foliummap.png)
![foliummap2](/image/foliummap2.png)



## Pumps & locations by Basemap 
![func](/image/func.PNG)
![need_repair](/image/need_repair.PNG)
![non_func](/image/non_func.PNG)



# 5. Feature engneering

**New features**
* population_zero :Categorised : population is  zero or not zero
* payment_status : Categorised : payment status is paid, not paid or unknown
* construction_year_new : Sort years by decades
* installer_group : Keep top 10 of installer
* funder_group :  Keep top 10 of funder

**Missing data**
* longitude : Missing value filled with median
* gps_height : Missing value filled with median



# 6. Modeling and Evaluation

## 6.1 Comparason of 4 Baseline Models's evaluation 
I have tried two different evaluation methods. Here it shows that the results are similar, I suppose it is becuase of the data size. 

**Datasets (without upsampling), holdout method (8:2) validation**

![baseline_model_scores](/image/baseline_model_scores.PNG)

**Datasets (without upsampling), 5 folds cross validation**

![baseline_model_scores_x_val](/image/baseline_model_scores_x_val.PNG)


**Upsampled Datasets, 5 fold cross validation, turned hyperparameters**

![upsampled_x_val_turned](/image/)



## 6.2  Classification report
I have decided to use XGB and optimise its hyperparameters because I would like to get to know this model. I am familiar with Random Forest Classifier from my other projects. Here it shows the details of XGB performance and it is turned by grid search. 


**Classification report XGB Baseline without upsampled train data**

![classification report baseline xgb without upsampling](/image/classification_report_xgb.PNG)

**Classification report Turned XGB model trained by upsampled data**


![Classification report Turned XGB model with upsampled train data](/image/)
# !!!!!!!!!! Turned version is not higher score than baseline. DO randomise search instead of grid search !!!!!!!!!!!!!!!



## 6.3 Confusion matrix

**Turned XGB with resampled**

![xgb confusion matrix](/image/confusion_matrix_xgb.PNG)

## 6.4 AUC and ROC chart 

**Turned XGB with resampled**

![roc_auc_chartxgb](/image/auc_roc_chart_xgb.PNG)



## 6.5 The competition and scores

DrivenData works on projects at the intersection of data science and social impact, in areas like international development, health, education, research and conservation, and public services. 
In this competition, **5600 competitors** joined and the best public score (30 Oct 2020) is **0.8294**. **My private score is 0.87.** [the website of the competion](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)
![competition](/image/competition.png)



# 8. Challenges and augmentations

* Many features(42), multi labels(3) and different kinds of data types(categorical, discrete, continous variables) makes this project difficult to resolve.
* Randomised and Grid serach took a while to compute optimal parameters due to the size of the dataset. I used Colab to generate the results faster than my local environment.
* Label 1 (needs repair) is extremely small percentage (0.07%). Even after increasing samples by SMOTE, the final model's weakness is detecting this class although it improved a lot comapared to a result without SMOTE. I will continue to look for a solution to improve. 
* I have done wrong process of Upsampling + Kfold Cross Validation on my train sets which returned extremely high results at first attempt. I have realised that some parts of upsampled X and y train datasets were copied of X and y train, theredore, this ended up overfitting by memorizing its training set. This was why cross validation scores were much higher than non upsampled train datasets' socres. Right steps are :
    - step 1 Oversample the minority class
    - step 2 Train the classifier on the training folds
    - step 3 Validate the classifier on the remaining fold


* I will continue to work on this project and would like to create more new features using missing features which are *not missing at random*. 
* Train data's features I used and created were not suitable for joining the competition because I used train label information while processing and creating new features. My next step by using this dataset is to create a model which can classify "test-only" test dataset and join a competion. 










