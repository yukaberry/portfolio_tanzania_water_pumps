# Project : Data Science Classification Problem
# Detect faulty water pumps in Tanzania

The objective of this project: To pridect which water pumps are functional, which need some repairs, and which don't work by using given dataset from the competition platform. [Here to find out more about the project report](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/findings/analysis.md), which contains below. 

1. The objecttive of this project
2. Data details
3. Domain 
4. Exploring and cleaning data
5. Feature engneering
6. Modeling and evaluation
7. The competition and final scores
8. Challenges and augmentations

## See interactive map on nbviewer [https://nbviewer.jupyter.org/github/yukaberry/portfolio_tanzania_water_pumps/blob/master/folium_map_layercontrol.ipynb]

![foliummap](image/foliummap.png)
![foliummap2](image/foliummap2.png)
![basemap](image/func.PNG)


DrivenData works on projects at the intersection of data science and social impact, in areas like international development, health, education, research and conservation, and public services. 
In this competition, **5600 competitors** joined and the best public score (30 Oct 2020) is **0.8294**. **My private score is 0.88.** [the website of the competion](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)
![competition](image/competition.png)




# Motivation

Interested in processing **geospatial data** and taking my understandings to a next level to deepen data science knowledge and skills. This problem has both categorical and numeric features, and multi-labels. This project spotlights **data visualisaiton** as well as **preprocessing and XGBoost modeling**. 


# Files' and folders' descriptions
* [findings](https://github.com/yukaberry/portfolio_tanzania_water_pumps/tree/master/findings) **The project folder**
  - [analysis.md](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/findings/analysis.md) **The summary and details of this project.**
* [data](https://github.com/yukaberry/portfolio_tanzania_water_pumps/tree/master/data) A data folder containing raw and processed data files
* [image](https://github.com/yukaberry/portfolio_tanzania_water_pumps/tree/master/image) Images' folder used for [analysis.md]()
* [model]() A folder of a picked model
* [utils](https://github.com/yukaberry/portfolio_tanzania_water_pumps/tree/master/utils) Utils folder for data cleaning, feature engneering and resampling. 
  - [cleaning_data_utils.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/utils/cleaning_data_utils.py) 
  - [feature_engneering_utils.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/utils/feature_engneering_utils.py)
  - [resampling_utils.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/utils/resampling_utils.py)
  - [modeling_utils.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/utils/modeling_utils.py)
* [requirements.txt](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/requirements.txt) Software Requirements

* [folium_map_layercontrol.ipynb](https://nbviewer.jupyter.org/github/yukaberry/portfolio_tanzania_water_pumps/blob/master/folium_map_layercontrol.ipynb) **Interactive water pumps' maps of Tanzania** in python notebook by using folium and geospatial data.
* [basemap.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/basemap.py) Maps plotted by basemap
* [data_visualisation.ipynb](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/data_visualisation.ipynb) Data visualisation codes and images in python notebook
* [cleaning_data.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/cleaning_data.py) Data cleaning and feature engneering.
* [resampling.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/resampling.py) Impliment a resampling technique.
* [modeling.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/modeling.py) Building models.
* [modeling_gridsearch.py](https://github.com/yukaberry/portfolio_tanzania_water_pumps/blob/master/modeling_gridsearch.py) Turning XGBoost model's parameters and it generates cross validation score.

  

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
