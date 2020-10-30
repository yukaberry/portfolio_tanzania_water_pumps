import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from utils.cleaning_data_utils import Scaling
from utils.cleaning_data_utils import return_median 
from utils.cleaning_data_utils import replace_nan_with_zero 

from utils.feature_engneering_utils import installer_cl
from utils.feature_engneering_utils import funder_cl
from utils.feature_engneering_utils import year_cl
from utils.feature_engneering_utils import pop_cl
from utils.feature_engneering_utils import pay_cl



# read csv
df = pd.read_csv("data/Training_set_values.csv")
label = pd.read_csv("data/training_set_labels.csv")

# merge two datasets
df = pd.merge(df,label, on = "id")

# making a new numeric label for classification model
le = preprocessing.LabelEncoder()
le.fit(df["status_group"])
df["label"] =le.transform(df["status_group"])


# Longitude : missing value filled with median
replace_nan_with_zero(df,"longitude")
return_median(df,"longitude")

# Save 
save_it_to_variable = return_median(df,"longitude")

# replace median values with zero value
df.loc[(df['status_group'] == "functional" ) & (df["longitude"].isnull()), "longitude"] = save_it_to_variable[0]
df.loc[(df['status_group'] == "functional needs repair" ) & (df["longitude"].isnull()), "longitude"] = save_it_to_variable[1]
df.loc[(df['status_group'] == "non functional" ) & (df["longitude"].isnull()), "longitude"] = save_it_to_variable[2]


# gps_height : missing value filled with median
replace_nan_with_zero(df,"gps_height")
return_median(df,"gps_height")

# Save 
save_it_to_variable_gps = return_median(df,"gps_height")

# replace median values with zero value
df.loc[(df['status_group'] == "functional" ) & (df["gps_height"].isnull()), "gps_height"] = save_it_to_variable_gps[0]
df.loc[(df['status_group'] == "functional needs repair" ) & (df["gps_height"].isnull()), "gps_height"] = save_it_to_variable_gps[1]
df.loc[(df['status_group'] == "non functional" ) & (df["gps_height"].isnull()), "gps_height"] = save_it_to_variable_gps[2]

# Keep top 10 of funder and installer
df['installer_group'] = df.apply(lambda df: installer_cl(df), axis=1)
df['funder_group'] = df.apply(lambda df: installer_cl(df), axis=1)


# Sorting decades
df['construction_year_new'] = df.apply(lambda df: year_cl(df), axis=1)

# new columns tells if population is  zero or not zero
df["population_zero"] = df.apply(lambda df:pop_cl(df),axis =1)

# new columns tells if payment status is paid, not paid or unknown
df["payment_status"] = df.apply(lambda df:pay_cl(df),axis =1)



# Feature scaling : standarisaton
# list of numeric variables 
columns_list = ["gps_height","population","longitude","latitude"]
Scaling(df,columns_list)

# Remove unnecessary columns 
to_drop = ["funder","installer","payment","payment_type","wpt_name","recorded_by","num_private","extraction_type",
           "extraction_type_group","scheme_management","scheme_name","management",
          "quality_group","quantity_group","source","public_meeting","lga","ward",
           "subvillage","region_code","district_code","date_recorded","id",'status_group'
           ,"waterpoint_type_group","permit","construction_year","amount_tsh"]
df.drop(to_drop,inplace= True,axis =1)

# transform Categotical variable to numeric variable
df =pd.get_dummies(df,columns=["basin","region","extraction_type_class","management_group","payment_status","water_quality","quantity","source_type","source_class",
                                            "waterpoint_type","installer_group","construction_year_new","funder_group","population_zero"
                                           ])

df = df[['gps_height','longitude','latitude','population','basin_Internal',
 'basin_Lake Nyasa','basin_Lake Rukwa', 'basin_Lake Tanganyika',
 'basin_Lake Victoria', 'basin_Pangani', 'basin_Rufiji',
 'basin_Ruvuma / Southern Coast', 'basin_Wami / Ruvu', 'region_Arusha',
 'region_Dar es Salaam', 'region_Dodoma', 'region_Iringa', 'region_Kagera',
 'region_Kigoma', 'region_Kilimanjaro', 'region_Lindi', 'region_Manyara',
 'region_Mara', 'region_Mbeya', 'region_Morogoro', 'region_Mtwara',
 'region_Mwanza', 'region_Pwani', 'region_Rukwa' ,'region_Ruvuma',
 'region_Shinyanga', 'region_Singida', 'region_Tabora', 'region_Tanga',
 'extraction_type_class_gravity', 'extraction_type_class_handpump',
 'extraction_type_class_motorpump', 'extraction_type_class_other',
 'extraction_type_class_rope pump' ,'extraction_type_class_submersible',
 'extraction_type_class_wind-powered', 'management_group_commercial',
 'management_group_other', 'management_group_parastatal',
 'management_group_unknown' ,'management_group_user-group',
 'payment_status_NeverPay', 'payment_status_Pay', 'payment_status_Unknown',
 'water_quality_coloured', 'water_quality_fluoride',
 'water_quality_fluoride abandoned', 'water_quality_milky',
 'water_quality_salty', 'water_quality_salty abandoned',
 'water_quality_soft', 'water_quality_unknown', 'quantity_dry',
 'quantity_enough', 'quantity_insufficient', 'quantity_seasonal',
 'quantity_unknown', 'source_type_borehole', 'source_type_dam',
 'source_type_other' ,'source_type_rainwater harvesting',
 'source_type_river/lake', 'source_type_shallow well', 'source_type_spring',
 'source_class_groundwater', 'source_class_surface', 'source_class_unknown',
 'waterpoint_type_cattle trough', 'waterpoint_type_communal standpipe',
 'waterpoint_type_communal standpipe multiple', 'waterpoint_type_dam',
 'waterpoint_type_hand pump', 'waterpoint_type_improved spring',
 'waterpoint_type_other' ,'installer_group_central gov',
 'installer_group_commu', 'installer_group_danida', 'installer_group_dwe',
 'installer_group_gov', 'installer_group_hesewa', 'installer_group_kkkt',
 'installer_group_others', 'installer_group_rwe', 'installer_group_tcrs',
 'installer_group_unknown' ,'construction_year_new_00s',
 'construction_year_new_10s', 'construction_year_new_60s',
 'construction_year_new_70s', 'construction_year_new_80s',
 'construction_year_new_90s', 'construction_year_new_unknown',
 'funder_group_central gov', 'funder_group_commu' ,'funder_group_danida',
 'funder_group_dwe', 'funder_group_gov', 'funder_group_hesewa',
 'funder_group_kkkt', 'funder_group_others', 'funder_group_rwe',
 'funder_group_tcrs', 'funder_group_unknown', 'population_zero_0',
 'population_zero_1','label']]

# Feature selection 
X = df.iloc[0:59400,0:110]
y= df.iloc[0:59400,-1]

df.to_csv("data/tanzania_cleaned_df.csv",encoding='utf8',index=False)
df = pd.read_csv("data/tanzania_cleaned_df.csv")

# MOMO
# do something with missing data ( gps, popluration, water amount)



