# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:49:03 2022

@author: GomezOspina.M

"""
# =============================================================================
# Import libraries
# =============================================================================

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import data
# =============================================================================

GW_CD_ID= gpd.read_file("./GIS/SHP/GWL_CDID.shp")
GW_CD_sel=GW_CD_ID[GW_CD_ID.KLIGL_GRUP==1]

def readGWdata(mestid):
    GWL=pd.read_csv("./Grundwasserstandsdaten/Einzelmessstellen/"+str(mestid)+".txt",parse_dates=['DATUM'], sep=";", decimal=".",header=0 )
    return GWL


tval=[]
c=0
for gid in GW_CD_sel.MEST_ID : 
    
    gw=readGWdata(gid)
    plt.plot(gw.DATUM, gw.GW_NN+c)
    tval.append(gw.GW_NN.count())
    c+=10

    
val=len(np.where(np.array(tval)>400)[0])   
plt.plot(tval)
plt.axhline(400)