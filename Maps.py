# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:07:33 2022

@author: mgome
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import os
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
#from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx

from shapely.geometry.point import Point

rpath="D:/Erasmus/Thesis/SomeData" #Root path


#Import Groundwatwer stations
GW_CD_ID= gpd.read_file(rpath+"/SHP/GWL_CDID.shp")
GW_CD_sel=GW_CD_ID[GW_CD_ID.KLIGL_GRUP.isin([1,12,13])] # selection of stations in good agreement with climatic variables

#%%
#Plot GW stations with data information
germany_states = gpd.read_file(rpath+"/SHP/DEU_adm1.shp")
NS=germany_states[germany_states.NAME_1== "Niedersachsen"]
GW_CD_ID=GW_CD_ID.to_crs(epsg=4326)
GW_CD_sel=GW_CD_sel.to_crs(epsg=4326)
germany=germany_states.boundary.plot(figsize=(15,15), edgecolor='k', lw=0.2)
NS_plot=NS.boundary.plot(figsize=(15,15), edgecolor='k', lw=0.7, ax=germany)
GW_CD_ID.plot(ax=NS_plot,marker='v', color='cadetblue', markersize=5, label="GW")
# GW_CD_sel.plot(ax=NS_plot,markersize=25, marker="o",facecolor="None", 
#                 edgecolor="darkred", label="GW_sel", alpha=0.5)
GW_CD_sel.plot(ax=NS_plot,markersize=10, marker="v",facecolor="None", 
                edgecolor="darkred", label="GW_sel", alpha=0.8)


limx=6.3
limy=51.2
germany.set_xlim(6.3, 12.8)
germany.set_ylim(51.2, 54)
germany.tick_params(axis='y', which='major', labelsize=10, rotation=90)
germany.tick_params(axis='x', which='major', labelsize=10, rotation=0)
startx, endx = germany.get_xlim()
starty, endy = germany.get_ylim()
stepsizex=1
stepsizey=0.5
germany.xaxis.set_ticks(np.arange(startx, endx, stepsizex))
germany.yaxis.set_ticks(np.arange(starty+.4, endy, stepsizey))


#Location map
locbox=inset_axes(NS_plot, width="40%", height='40%',loc='lower right', borderpad=0, \
                  bbox_to_anchor=(9.2,51.2,3.5,3.5),  bbox_transform=NS_plot.transData)
germany_states.boundary.plot(figsize=(20,20),edgecolor='k', lw=0.2,ax=locbox)
NS.plot(color="red", alpha=0.3, ax=locbox)
NS.boundary.plot(edgecolor='red', color="red", alpha=0.3, lw=0.7, ax=locbox)
locbox.tick_params(axis='y', which='major', labelsize=7, rotation=90)
locbox.tick_params(axis='x', which='major', labelsize=7, rotation=0)
startx, endx = locbox.get_xlim()
starty, endy = locbox.get_ylim()
stepsizex=4
stepsizey=3
locbox.xaxis.set_ticks(np.arange(startx+1, endx, stepsizex))
locbox.yaxis.set_ticks(np.arange(starty+1, endy, stepsizey))
locbox.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
locbox.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))


#North arrow
arrx=11.9
arry=53.8
germany.text(x=arrx, y=arry, s='N', fontsize=20,alpha=0.8)
germany.arrow(arrx+0.08, arry-0.05, 0, 0.01, length_includes_head=True,
          head_width=0.17, head_length=0.25, overhang=.2, ec="k",facecolor='k', alpha=0.4)


#ctx.add_basemap(germany, crs = germany_states.crs, url = ctx.providers.Esri.WorldShadedRelief)


#GW_CD_ID.plot(ax=NSmap,marker='*', color='c', markersize=8, label="GW")
# NSmap.plot(column='nonan_yr',scheme="Quantiles", markersize=GW_CD_sel.nonan_yr.values/5,
#          legend=True, label="GW_sel")

