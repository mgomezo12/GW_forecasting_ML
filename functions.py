# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:01:00 2022

@author: GomezOspina.M
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import netCDF4 as nc
from datetime import datetime, timedelta
import glob

#Read groundwater time series (txt)
def readGWdata(mestid, path):
    GWL=pd.read_csv(path+str(mestid)+".txt",parse_dates=['DATUM'], sep=";", decimal=".",header=0 )
    return GWL


def netcdfFiles(folderpath):
    "Read all netCDF files in a folder"
    files = glob.glob(folderpath+'/*.nc')
    netFiles = nc.MFDataset(files) 
    return netFiles


class netcdfdata:
    """Method to extract information from netCDF data given a shapefile of points 
    ________________________________________________
    Parameters: 
        file: netCDF file 
        lat, lon : standard latitude and longitude 
                    (if the netCDF only has rotated polar coordinates since 
                     the transformation is not completely matching)
        
   """
   
    def __init__(self, file, shp, lat=None ,lon=None):
        self.olon= file.variables["rotated_pole"].grid_north_pole_longitude
        self.olat= file.variables["rotated_pole"].grid_north_pole_latitude
        self.data= file
        self.ltime=len(file.variables["time"][:])
        self.tlon= shp.geometry.x
        self.tlat= shp.geometry.y
        
        if lat or lon is None:
            self.lon= self.data.variables['lon'][:]
            self.lat= self.data.variables['lat'][:]
        else:
            self.lon=lon
            self.lat=lat
        

    def transform_rot_coord(self):
        "This function is under revision \
        It is thought for projecting rotated polar coordinates to standard lat/lon coordinates \
        in particular, from the HYRAS dataset from the DWD Germany"
        al=list(self.data.variables.keys())
        if any("lat" in s for s in al)==False :
            trans = Transformer.from_crs('epsg:4326', '+proj=ob_tran +o_proj=longlat \
                                         +o_lon_p='+str(self.olon) + '+o_lat_p='+str(self.olat) +' +lon_0=180', always_xy=True)
                                         
            rlon = self.data.variables['rlon'][:]
            rlat = self.data.variables['rlat'][:]
            lon, lat = [] ,[]
            for lo, la in zip(rlon[:], rlat[:]):
                lonv, latv= trans.transform(lo,la)
                lon.append(lonv)
                lat.append(latv) 
            return lon, lat
    

    def tlonlat(self):
        """Give the index of the target longitude and latitude  """
        
        vtlon=[]
        vtlat=[]
        for lonval,latval in zip(self.tlon,self.tlan): 
            ji = np.sqrt( (self.lon-lonval)**2 + (self.lat-latval)**2 ).argmin()
            tarlon,tarlat = np.unravel_index(ji, self.lon.shape)
            vtlon.append(tarlon)
            vtlat.append(tarlat)
        
        return vtlon, vtlat
    
    def extractTS(self):
        """Extract the climate data in a domain of 3x3 pixels 
        as recommended by DWD
        """ 
        
        llon=self.tlonlat(self.tlon,self.tlat)[0]
        llat=self.tlonlat(self.tlon,self.tlat)[1]
        
        cdts=[]
        cdts_list=[]
        for j, i in zip(llon,llat):
            for n in range(self.ltime):
                cdts.append(self.data['hurs'][n,j-1:j+2,i-1:i+2].mean())
            cdts_list.append(cdts)
        cd_dic={"lon":llon, "lat": llat, "cdata":cdts_list}
        cd_data=pd.DataFrame(cd_dic)
        
        return cd_data
        

GWF = gpd.read_file("C:/Users/GomezOspina.M/MGO/data/GIS/SHP/GWF.shp")

    
    