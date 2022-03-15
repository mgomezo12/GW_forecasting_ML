# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:01:00 2022

@author: GomezOspina.M
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Read groundwater time series (txt)
def readGWdata(mestid, path):
    GWL=pd.read_csv(path+str(mestid)+".txt",parse_dates=['DATUM'], sep=";", decimal=".",header=0 )
    return GWL
