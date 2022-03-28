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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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

        
class fillGWgaps:
    """ This class aim to fill the missing values in the groundwater level
    time series considering the neighbouring wells until a threshold distance. 
    A multiple linear regression is performed for a target well
    Wells with a defined percentage of no nan values are considered. 
    The no nan values in the non-target well are filled with the mean value of the corresponding series.
    ____________________________________
       
    Inputs:
        gwdata: dataframe with the wellids ("wellid"), data and "max_gap" (maximum monthly gap) column
        gws: shapefile of points with the well coordinates
        
        maxd: maximum fixed distance in meters to look for closest wells (int)
        th: % threshold of nan values, meaning that the column should 
                contain at least th% no NaN values to be consider            
        maxn: maximum number of wells near the twell to consider 
                for the MLR. Type:int
        
    Outputs: 
        
    
    """
    def __init__(self, gwdata, gws,maxd=2*10e3,th=98, maxn=8):
        self.gwdata=gwdata
        self.gws= gws
        self.maxd=maxd
        self.th=th
        self.maxn=maxn
        
    def distmat(self):
        "Generate distance matrix based on the well coordinates"
        gdfs = gpd.GeoDataFrame(geometry=self.gws.geometry).to_crs("EPSG:4647")
        dmtca=gdfs.geometry.apply(lambda g: gdfs.distance(g))
        dmtca.columns=self.gws["MEST_ID"].apply(str)
        dmtca.index=self.gws["MEST_ID"]
        
        return dmtca
    
    def nearwells(self):
        """Generates a dataframe with the nearest wells to each well up 
        to a maximum fixed distance (maxd in meters)"""
        idv=[]
        dmtca=self.distmat()
        ids=dmtca.columns[:]
        val=[]
        for i in ids:
            sort_dm=dmtca[i].sort_values()
            dmtca_bol=sort_dm < self.maxd
            idlist=dmtca_bol==True
            values=sort_dm.loc[dmtca_bol == True].values/1000 #km
            idv.append(idlist.loc[dmtca_bol == True].index)
            val.append(values)
        dic_id={"wellid": ids, "nearwell":idv, "Distance":val}
        idnear=pd.DataFrame(dic_id)
        
        return idnear
    
    def tw_data(self, twell):
        """
        Parameters
        ----------
        twell : target well
        
        Returns
        -------
        dfwell : Dataframe  with the data of the target well
        """
        iwell=self.gwdata.data[self.gwdata["wellid"]==twell].index[0]
        data_twell=self.gwdata.data[self.gwdata["wellid"]==twell]
        dfwell=data_twell[iwell] 
        
        #List of wells near the target well
        # Exclude the first well since it's the same target well
        lwells=self.nearwells().nearwell[self.nearwells()["wellid"]==str(twell)].values[0][1:]
        
        return dfwell, lwells
    
    def builttwdf(self,twell):
        """
        Generates a Dataframe based on the dates of the target well and 
        merge the information with the nearest wells 
        -------
        input: twell --> code of the target well (the one to be gap-filled)
                type: int

    
        """       
        dfwell=self.tw_data(twell)[0]
        lwells=self.tw_data(twell)[1]
        
        #Begining and end of the time series
        #Where no sequential NaN values are present
        fnonan=dfwell.GW_NN.first_valid_index()
        lastnonan=dfwell.GW_NN.last_valid_index()
        
        filldf=pd.DataFrame({"DATUM":dfwell.DATUM.loc[fnonan:lastnonan]})
        for w in lwells:
            series=self.gwdata.data[self.gwdata["wellid"]==w] 
            if not series.empty: # Remove wells with no information
                #ask for data at the wellid
                wellind=self.gwdata.data[self.gwdata["wellid"]==w].index[0]
                welldata=self.gwdata.data[self.gwdata["wellid"]==w][wellind]
                
                #join data into the filldf dataframe
                mergedf=pd.merge(welldata[["DATUM","GW_NN"]],filldf, on=["DATUM"])
                mergedf.rename(columns={"GW_NN":"GW_NN_"+str(w)}, inplace= True)
                
                filldf=mergedf
        
            else:
                print("empty")
        
        if not filldf.empty:
            filldf["twell_"+str(twell)]=dfwell.GW_NN.loc[fnonan:lastnonan].values
            
            
            #Training dataset
            thresh=int(len(filldf)*self.th/100) 
            nonanmerge=filldf.dropna(axis=1, thresh=thresh)
            
            nonanmerge2=nonanmerge.copy()
            col=nonanmerge2.columns
            #fill NaN in the nearby wells with the mean value of the time series of the well
            fillednan=nonanmerge2[col].fillna(nonanmerge2[col].mean()) 
            fillednan[mergedf.columns[-1]]=mergedf[mergedf.columns[-1]]
            dftest=fillednan.dropna()
        
        return dftest, fillednan
    
    def MLRmodel(self, twell):
        """ Multiple linear regression
        twell --> code of the target well (the one to be gap-filled)
                type: int
                
        maxn:  maximum number of wells near the twell to consider 
        for the MLR. Type:int
        """
        
        dftest=self.builttwdf(twell)[0]
        fillednan=self.builttwdf(twell)[1]
        
        #Define dataset
        auxdf=dftest[dftest.columns[:-1]] #Exclude twell column
        X = auxdf[auxdf.columns[1:self.maxn+1]]
        y = dftest[dftest.columns[-1]]
        
        #TEST and TRAIN
        if not X.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y,\
                                                        test_size = 0.20,\
                                                        random_state = 5)
            model1 = LinearRegression().fit(X_train, y_train)
            score=model1.score(X_test, y_test)
            
            predictdf=fillednan[fillednan["twell_"+str(twell)].isna()]
            auxdf2=predictdf[predictdf.columns[:-1]] #Exclude twell column
            X_predct=auxdf2[auxdf2.columns[1:self.maxn+1]]
            predictions = model1.predict(X_predct)
            predictdf["twell_"+str(twell)]=predictions
            dftest2=dftest.combine_first(predictdf[["DATUM", "twell_"+str(twell)]])
            
            return dftest2 , score 
        else:
            print("No wells avaialble for model prediction")
            
    
    
        
        
        
        
    
    