# -*- coding: utf-8 -*-
"""
mgomezo12

"""
# =============================================================================
# Import libraries
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd
import contextily as cx

#%%

# =============================================================================
# Import data
# =============================================================================

#paths
path=r"./Klimabeobachtungen/"
path2=r"./Grundwasserstandsdaten/"
pathGIS=r"./GIS/"

##Draft 
#Load climate stations 
cd=pd.read_excel(path + "KL_Stationen_Info.xls") # Climate data stations
rr=pd.read_excel(path + "RR_Stationen_Info.xls") #Rainfall data stations

#Load groundwater stations
gw=pd.read_csv(path2 + "PROJEKT_BASISDATEN.txt",sep=";", decimal=",",header=0) 

#%%

GW_CD_ID1= gpd.read_file(pathGIS+"./SHP/GWL_CDID.shp")# GW associated to the climate model -- Raster pixels


germany_states = gpd.read_file(pathGIS+"./SHP/DEU_adm1.shp")
NS=germany_states[germany_states.NAME_1== "Niedersachsen"]
clst = gpd.GeoDataFrame(cd,geometry=gpd.points_from_xy(cd.LAENGE, cd.BREITE))
rrst = gpd.GeoDataFrame(rr,geometry=gpd.points_from_xy(rr.LAENGE, rr.BREITE))
GWst = gpd.GeoDataFrame(gw,geometry=gpd.points_from_xy(gw.UTM_RECHTS, gw.UTM_HOCH))
GWst = GWst.set_crs(epsg=4647)
GWstc= GWst.to_crs(epsg=4326)
GW_CD_ID1=GW_CD_ID1.to_crs(epsg=4326)
#%%
#ax =germany_states.boundary.plot(figsize=(10, 10),alpha=0.5, edgecolor='k', linewidth=1)
NSmap= NS.boundary.plot( figsize=(10, 10), alpha=0.5, edgecolor='k', linewidth=1)
clst.plot(ax=NSmap,marker='v', color='darkred', markersize=5, label="CD")
rrst.plot(ax=NSmap,marker='.', color='c', markersize=5, label="RR")
#GWstc.plot(ax=NSmap,marker='o', color='chocolate', markersize=8, label="GWCD")
GW_CD_ID1.plot(ax=NSmap,marker='*', color='darkblue', markersize=2, label="GW")

plt.legend()
#cx.add_basemap(NSmap)









