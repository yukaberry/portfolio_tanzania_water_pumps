import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from sklearn import preprocessing

# read and merge csv
data = pd.read_csv("/kaggle/input/tanzania-water-pumps/train.csv")
lable =pd.read_csv("/kaggle/input/tanzania-water-pumps/label.csv")
df = pd.merge(data,lable, on = "id")

# create dataset for functional, need repair, non functional
d_func = df.loc[lambda df: df["status_group"] =="functional"]
d_repair = df.loc[lambda df: df["status_group"] =="functional needs repair"]
d_n_func = df.loc[lambda df: df["status_group"] =="	non functional"]

# Preparing data for plotting
# Data needs to be list or array
lons_repair = d_repair["longitude"]
lats_repair = d_repair["latitude"]

lats_repair = lats_repair.values.tolist()
lons_repair = lons_repair.values.tolist()

lons_n_func = d_n_func["longitude"].values
lats_n_func = d_n_func["latitude"].values

lons_func = d_func["longitude"].values
lats_func = d_func["latitude"].values

# Set up "alpha=0.1" to see density of need-repair pumps
# Basema for needs repair
fig = plt.figure(figsize = (12,9))
m = Basemap(projection ="mill",
            llcrnrlat = -13,
           urcrnrlat =1,
            llcrnrlon = 28,
            urcrnrlon = 44,
           resolution = "l")
m.drawcountries(color= "brown",linewidth =2)
plt.title("Tanzania Faulty Water pumps : Need repair", fontsize = 20)
m.drawcoastlines()
m.etopo()

x,y = m(lons_repair,lats_repair)
m.scatter(x, y,color ="red", alpha=0.1)
plt.show()


# Basemap for functional 
fig = plt.figure(figsize = (12,9))
m = Basemap(projection ="mill",
           llcrnrlat = -13,
           urcrnrlat =1,
            llcrnrlon = 28,
            urcrnrlon = 44,
           resolution = "l")
m.drawcountries(color= "brown",linewidth =2)
plt.title("Tanzania Faulty Water pumps : Non Functional", fontsize = 20)
m.drawcoastlines()
m.etopo()

x,y = m(lons_n_func,lats_n_func)
m.scatter(x, y,color ="pink",alpha=0.1)
plt.show()


# Basemap for non functional
fig = plt.figure(figsize = (12,9))
m = Basemap(projection ="mill",
           llcrnrlat = -13,
           urcrnrlat =1,
            llcrnrlon = 28,
            urcrnrlon = 44,
           resolution = "l")
m.drawcountries(color= "brown",linewidth =2)
plt.title("Tanzania Faulty Water pumps : Functional", fontsize = 20)
m.drawcoastlines()
m.etopo()

x,y = m(lons_func,lats_func)
m.scatter(x, y,color ="lightgreen",alpha=0.1)
plt.show()




