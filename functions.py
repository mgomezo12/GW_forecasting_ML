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
import datetime as dt

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
        shp: shapefile points to extract the climate information
        variable: the variable name according to the dataset
        
        lat, lon : standard latitude and longitude 
                    (if the netCDF only has rotated polar coordinates since 
                     the transformation is not completely matching)
        
        
   """
   
    def __init__(self, file, shp, variable="pr", lat=None ,lon=None):
        self.olon= file.variables["rotated_pole"].grid_north_pole_longitude
        self.olat= file.variables["rotated_pole"].grid_north_pole_latitude
        self.data= file
        self.time=file.variables['time']
        self.ltime=len(file.variables["time"][:])
        self.tlon= shp.geometry.x
        self.tlat= shp.geometry.y
        self.GWID= shp.MEST_ID
        self.variable=variable

        
        if isinstance(lat,np.ma.core.MaskedArray):
            self.lon=lon
            self.lat=lat            
        else:
            self.lon= self.data.variables['lon'][:]
            self.lat= self.data.variables['lat'][:]
        

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
        for lonval,latval in zip(self.tlon,self.tlat): 
            ji = np.sqrt( (self.lon-lonval)**2 + (self.lat-latval)**2 ).argmin()
            tarlon,tarlat = np.unravel_index(ji, self.lon.shape)
            vtlon.append(tarlon)
            vtlat.append(tarlat)
        
        return vtlon, vtlat
    
    def extractTS(self):
        """Extract the climate data in a domain of 3x3 pixels (mean value)
        as recommended by DWD
        """ 
        
        llon=self.tlonlat()[0]
        llat=self.tlonlat()[1]
        
        
        cdts_list=[]
        for j, i in zip(llon,llat):
            cdts=[]
            for n in range(self.ltime):
                cdts.append(self.data.variables[self.variable][n,j-1:j+2,i-1:i+2].mean())
            print(j,i)
            #print(cdts)
            cdts_list.append(cdts)
        
        #Datetime dataset
        units=self.time.units
        calendar=self.time.calendar
        timearray=np.arange(self.ltime)
        dtime=nc.num2date(timearray, units, calendar)
        d1=dtime[:].astype(str)
        dates_list = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').date() for date in d1]
        
        cd_dic={"ID":self.GWID,"lon":llon, "lat": llat, "time":[dates_list]*len(llon), "cdata":cdts_list}
        cd_data=pd.DataFrame(cd_dic)
        
        return cd_data

        


    
    