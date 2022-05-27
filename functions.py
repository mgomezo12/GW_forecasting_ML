# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:01:00 2022

@author: GomezOspina.M
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pyproj import Transformer
import netCDF4 as nc

import glob
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress
import pymannkendall as mk

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal


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
            #print(j,i)
            #print(cdts)
            cdts_list.append(cdts)
        print(len(cdts_list))
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






class loadccvar:
    """This class is to help in the reading process of the climate model data
    given by the DWD Germnay (dataset downscaled and bias corrected)
    
    -------
    Inputs: 
        modelname (str): name of the climate model according to the files
        var (str): acronym for the climate variable
        varcodfile (str): short code of the climate variable associated with the file 
        cod (str): code associated with the each time series file
        vcod (pandas series or list): ID of points to retrieve the time series information
        scenario (str): the climate RCP projection to use 
        
        path containing the folders with the climate models
        
        output: dataframe containing the time series per well id, the trend slope and intercept
        
    """
    def __init__(self, modelname, var, varcodfile, cod, vcod, idcods, path, scenario="RCP85"):
        self.modelname=modelname
        self.var= var
        self.varcodfile=varcodfile
        self.cod=cod
        self.vcod=vcod
        self.idcods=idcods
        self.scenario=scenario
        self.rpath=path
        
    
    def mktrend(self,series, period=12, alpha=0.05, met="MK"):
        """"
        
        method (met): Mann Kendall method 
        Period:12 months, if the time-step resolution changes, 
        the period should be adjusted
        
        this function uses the seasonal Mann Kendall test
        developed by Hussain et al., (2019)
        
        """
        
        if met == "MK":
    
            trend, h, p, z, Tau, s, var_s, slope, intercept=mk.seasonal_test(series, 
                                                                             period=period, 
                                                                             alpha=alpha)
 #           trend, h, p, z, Tau, s, var_s, slope, intercept=mk.original_test(series, alpha=0.05)
            
            return trend, p, s, slope, intercept
            
        else:
            """Seasonal decompose by the additive method and linear regression of the seasonal
             trend to check the general trend """
            decomp=seasonal_decompose(series, model="additive", period=period)
            dtrend=decomp.trend
            dtrendc=dtrend.copy().dropna()
            dtrendc.index= np.arange(len(dtrendc))
            x =np.arange(len(dtrendc)-1, dtype=np.float64)
            y = dtrendc.values[:-1].astype('float32')
            slope, intercept, r_value, p, std_err = linregress(x, y)
            if p > 0.05:
                trend="False"
            else:
                trend="True"
            
            return  trend, p, r_value, slope, intercept
    
    
    def readtimeseries(self):
        path=self.rpath+   \
            self.modelname+"_"+self.scenario.upper()+      \
            "/"+self.modelname+"_"+self.var+"_bk_"+       \
            self.scenario.lower()+"/" 
            
        if not os.path.exists(path):
            path=self.rpath+  \
            self.modelname+"_"+self.scenario.upper()+ \
            "/"+self.modelname+"_"+self.var+"_"+  \
            self.scenario.lower()+"/" 
        
        vtm, vidcods, vslope, vintercept, vtrend, vpval =[] , [] , [], [], [], []
        c=0
        for mid in self.vcod: 
            tmc=pd.read_csv(path+ \
                            self.varcodfile+str(int(mid)).zfill(4)+"."\
                            +self.cod, \
                            decimal=".",header=0 )
            
        
            datasplit=tmc[tmc.columns[0]].str.split("    ", expand=True)
            splitdates=datasplit[0].str.split(expand=True)
            dates=pd.to_datetime(splitdates[0].astype(str)+' '
                                 +splitdates[1].astype(str)+' '
                                 +splitdates[2].astype(str))
            data=datasplit[1].copy()
            

    
            dfdata=pd.DataFrame({"dates":dates, "data":data.astype(float)})
            
            dfdatacopy=dfdata.set_index("dates").copy()
            #Monthly resample
            if self.var== "pr":
                datamonth=dfdatacopy.resample("M").sum()
            else:
                datamonth=dfdatacopy.resample("M").mean()
            

            trend, p, s, slope, intercept= self.mktrend(series=datamonth, period=12)
            
            vtm.append(datamonth)
            vidcods.append(self.idcods[c])
            vtrend.append(trend)
            vpval.append(p)
            vslope.append(slope)
            vintercept.append(intercept)
            c+=1
            
        dfcm= pd.DataFrame({"wellid":vidcods,"data":vtm, "trend":vtrend, 
                            "pval":vpval,"slope":vslope, "intercept":vintercept})
        
        return dfcm
            
           




        
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
    def __init__(self, gwdata, gws,maxd=20*10e3,th=98, maxn=8):
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
                MEST_ID, DATUM, GW_NN (main columns)
        lwells: list of the ids of closest wells near the target well
        
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
                
        Output: dftest--> dataframe of the target well and the nearest wells information
                        the data is cliped to start and end in the first and last valid value 
                        (no sequential NaN)
                        Non-Nan values associated with the target well are removed
                        
                        
                fillednan--> Dataframe of the target well and the nearest wells information
                            is associated only with the NaN and NoNaN values of the target well
                            
                twdata--> Dataframe with the target well information, contains the Nan and No NaN 
                        information associated with the target well. 
    
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
    
    def euclidean_distance(self, s1,s2):
        """"Compute the Euclidean distance between two time series - 
        before estimation, the series are translated, scaled and detrend """
        #Transform first
        series1=signal.detrend((s1-s1.mean())/s1.std())
        series2=signal.detrend((s2-s2.mean())/s2.std())    
    
        dist=np.sqrt(sum(pow(a-b,2) for a, b in zip(series1, series2)))
        
        return dist
    
    def datasetsel(self, twell, distthresh=60):
        """"Check the similarity with the Euclidean distance and remove the times
        series with distances above distthresh (int) """
        
        
        vdist, vwell=[], []
        dftest=self.builttwdf(twell)[0]
        fillednan=self.builttwdf(twell)[1]
        
        
        for c in dftest.columns[1:-1]:
            s1=dftest[dftest.columns[-1]]
            s2=dftest[c]
        
            dist=self.euclidean_distance(s1,s2)
            vwell.append(c)
            vdist.append(dist)
           # distdf=pd.DataFrame({"idwells":vwell, "distance":vdist})
            #sim=1/(1+dist)
        norm = [float(i)*100/max(vdist) for i in vdist]
        lim=np.percentile(np.array(norm),10) ## Take the distances bellow the 10th percentile
        if lim >80: # If greater than 80 then use the distance set
            lim=distthresh
        ind=np.where(np.array(norm)<=lim) 
        vwell=np.array(vwell)
        vwell2=np.append(vwell[ind], ["DATUM", "twell_"+str(twell)])
        dfstation=dftest[dftest.columns[dftest.columns.isin(vwell2)]]
        fillnan=fillednan[fillednan.columns[fillednan.columns.isin(vwell2)]]

        
        return dfstation, fillnan
        

    def pchip(self, data):
        """Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) to fill the missing values 
        that can not be filled by other methods-- this method should be only for small gaps"""
        datai = data.interpolate(method="pchip", order=2)
        
        return datai
        
              

    def MLRmodel(self, twell,threshscore=0.7):
        """ Multiple linear regression
        twell --> code of the target well (the one to be gap-filled)
                type: int
                
        maxn (int):  maximum number of wells near the twell to consider 
        for the MLR. 
        """
        
        dfstation=self.datasetsel(twell)[0]
        fillednan=self.datasetsel(twell)[1]
        
        nanratio=(len(fillednan[fillednan["twell_"+str(twell)].isna()])/
                  len(fillednan["twell_"+str(twell)]))*100
        numnanr=len(fillednan[fillednan["twell_"+str(twell)].isna()])
        
        if len(fillednan[fillednan["twell_"+str(twell)].isna()])==0:
            interp_type="complete"
            return fillednan[["DATUM", "twell_"+str(twell)]] , interp_type, nanratio, numnanr
        

        
        #Define dataset
        dfstationax=dfstation[dfstation.columns[1:-1]]
        X = dfstationax[dfstationax.columns[-self.maxn:]]
        y = dfstation[dfstation.columns[-1]]
        
        #TEST and TRAIN
        if not X.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y,\
                                                        test_size = 0.20,\
                                                        random_state = 5)
            model1 = LinearRegression().fit(X_train, y_train)
            score=model1.score(X_test, y_test)
            
            if score>threshscore:    
                predictdf=fillednan[fillednan["twell_"+str(twell)].isna()]
                auxdf2=predictdf[predictdf.columns[1:-1]] #Exclude twell column
                X_predct=auxdf2[auxdf2.columns[-self.maxn:]]
                predictions = model1.predict(X_predct)
                predictdf["twell_"+str(twell)]=predictions
                dftest2=dfstation.combine_first(predictdf[["DATUM", "twell_"+str(twell)]])
                interp_type="MLR"
                
                return dftest2, interp_type, nanratio, numnanr
            
            else:
                fillednan["Ftwell_"+str(twell)]=self.pchip(fillednan["twell_"+str(twell)])
                twdata=fillednan[["DATUM", "Ftwell_"+str(twell)]]
                interp_type="PCHIP"
                return twdata, interp_type, nanratio, numnanr

        else:
            fillednan["Ftwell_"+str(twell)]=self.pchip(fillednan["twell_"+str(twell)])
            twdata=fillednan[["DATUM", "Ftwell_"+str(twell)]]
            interp_type="PCHIP"
            return twdata, interp_type, nanratio, numnanr
        

            
class setinputdataset:
    """"This class is to set the input features per well according to the 
    pre-processing outputs, which are saved into series of dataframes
    
    inputs:
        wellid (str): the id of the wanted well 
        datagw : dataframe with the groundwater information after filtering and gap 
        filling, the required columns are wellid and GW_NN (groundwater levels records)
        This is the output of the pre-processing --Filling_gaps_GWL

    """
    def __init__(self,wellid, datagw):
        self.wellid= wellid
        self.datagw=datagw        

    def setgwdata(self):
        "Set a dataframe with the groundwater levels records of the wellid"
        dfgwl=self.datagw.loc[self.datagw.wellid==int(self.wellid)]
        indv=dfgwl.index.values[0]
        datfill=dfgwl.GW_NN[indv]
        try:
            dfwell=datfill[["DATUM","twell_"+str(self.wellid)]]
        except:
            dfwell=datfill[["DATUM","Ftwell_"+str(self.wellid)]]
            
        return dfwell
    
    def selmetdata(self,datapr, datatm, datarh):
        """Monthly resample of metereological data and extraction per wellid
        
        datapr, datatm, datarh: dataframe of information per well, contains the daily time series
        of each metereological variable (precipitation, temperature, relative humidity), 
        the required columns are "time" (time in the format of datetime),
        "cdata"(climate time series) and "ID" (well ids). 
        
        This is the output of the pre-processing--extract_netCDF"""
        
        metdata=[datapr,datatm,datarh]
        metnames=["pr","tm", "rh"]
        
        vmetdata=[]
        for metdf, metname in zip(metdata, metnames):
            df=metdf.loc[metdf.ID==int(self.wellid)]
            indexm=df.index.values[0]
            #set dates as a column
            df_day=pd.DataFrame({"dates":pd.to_datetime(df.time[indexm]),
                                       "cdata":df.cdata[indexm]} )
            df_dayc=df_day.copy()
            #resample to monthly resolution
            if metname=="pr":
                dfcdmonth=df_dayc.resample("M", on="dates").sum()
            else:
                dfcdmonth=df_dayc.resample("M", on="dates").mean()
                
            dfcdmonth['dates'] = dfcdmonth.index
            vmetdata.append(dfcdmonth)
            
        return vmetdata
    
    def setinputdata(self,datapr, datatm, datarh):
        """Set climate data into a dataframe containing the three 
        metereological variables and the GWL
        
        output: dataframe with the monthly time series of the groundwater levels and
        the metereological information associated to the well id"""
        
        #Check for the time range compatibility-- the time range is check against the precipitation time range, since
        #the HYRAS dataset is available for the same time-range, this is also valid for temperature and relative humidity
        dfwell=self.setgwdata()
        metdfwell=self.selmetdata(datapr, datatm, datarh)
        
        dateini=dfwell.DATUM[0]  if  dfwell.DATUM[0]>metdfwell[0].index[0] else metdfwell[0].index[0]
        datefin=metdfwell[0].index[-1] if dfwell.DATUM[len(dfwell)-1] > metdfwell[0].index[-1] else dfwell.DATUM[len(dfwell)-1]
        dates= pd.date_range(dateini,datefin, freq='M')
        
        dfwell["DATUM"]= [dfwell.DATUM[n].strftime("%Y-%m") for n in range(len(dfwell))]

        #Make sure the data has the same time range 
        dfwellsel=dfwell.loc[(dfwell.DATUM>=dateini.strftime("%Y-%m")) & (dfwell.DATUM<=datefin.strftime("%Y-%m"))]
        
        vmetdfwell=[]
        for df in metdfwell:
            vmetdfwell.append(df.loc[(df.dates>=dateini) & (df.dates<=datefin)])

        #Save a dataframe with the information per well (historic data)
        try:
            cdwell=pd.DataFrame({"dates":dates,"GWL": dfwellsel["twell_"+str(self.wellid)], "pr":vmetdfwell[0].cdata.values,
                                 "tm":vmetdfwell[1].cdata.values, "rh": vmetdfwell[2].cdata.values})
        except:
            cdwell=pd.DataFrame({"dates":dates,"GWL": dfwellsel["Ftwell_"+str(self.wellid)], "pr":vmetdfwell[0].cdata.values,
                                 "tm":vmetdfwell[1].cdata.values, "rh": vmetdfwell[2].cdata.values})
            
        return cdwell
    
    
    def setclimmodel(self,cmpr,cmtm,cmrh, modelname="MPI_WRF361H"):
        
        climmodels=[cmpr,cmtm,cmrh] 
        dfwell=self.setgwdata()
       
        
         #List of climate models available
        lmodels=["MPI_WRF361H", "MPI_CCLM", "MIROC_CCLM", 
                 "HadGEM_WRF361H", "ECE_RACMO_r12", "ECE_RACMO_r1"]
        lmodel=np.array(lmodels)
        #locate the model to match with the name on the argumments
        nd=np.where(lmodel==modelname)[0][0]
        
        vcmwell=[]
        for mod in climmodels:
            modeldat=mod.cmodel[nd]
            cwmwells=modeldat.loc[modeldat.wellid==int(self.wellid)]
            icmwells=cwmwells.index.values[0]
            cmwell=cwmwells.data[icmwells]
            vcmwell.append(cmwell)
           
            
        dateini=dfwell.DATUM[0]  if  dfwell.DATUM[0]>vcmwell[0].index[0] else vcmwell[0].index[0]
        datefin=vcmwell[0].index[-1] if dfwell.DATUM[len(dfwell)-1] > vcmwell[0].index[-1] else dfwell.DATUM[len(dfwell)-1]
        dates= pd.date_range(dateini,datefin, freq='M')
        
        dfwell["DATUM"]= [dfwell.DATUM[n].strftime("%Y-%m") for n in range(len(dfwell))]
        
         #Make sure the data has the same time range 
        dfwellsel=dfwell.loc[(dfwell.DATUM>=dateini.strftime("%Y-%m")) & (dfwell.DATUM<=datefin.strftime("%Y-%m"))]
        
        vcmwellclim=[]
        for df in vcmwell:
            vcmwellclim.append(df.loc[(df.index.strftime("%Y-%m")>=dateini.strftime("%Y-%m")) & (df.index.strftime("%Y-%m")<=datefin.strftime("%Y-%m"))])
            

        
        #Save a dataframe with the information per well and the climate models
        try:
            cmwelldf=pd.DataFrame({"dates":vcmwellclim[0].index,
                                 "GWL": dfwellsel["twell_"+str(self.wellid)], 
                                 "pr":vcmwellclim[0].data.values,
                                 "tm":vcmwellclim[1].data.values, 
                                 "rh":vcmwellclim[2].data.values})
        except:
            cmwelldf=pd.DataFrame({"dates":vcmwellclim[0].index,
                                 "GWL": dfwellsel["Ftwell_"+str(self.wellid)], 
                                 "pr":vcmwellclim[0].data.values,
                                 "tm":vcmwellclim[1].data.values, 
                                 "rh":vcmwellclim[2].data.values})
        return cmwelldf 
        
   
    

def mapplot(data, gwstat, countrybd, column, namevar, units, axis,cmap):
    """This funtion is to create a map according to the input shapefiles
    
    data: geopandas dataframe (point shape) with the columns needed to classify
    gwstat: shapefile with the well locations
    countrybd: national or regional administrative boundaries.
    column: the name of the column to perform the classification
    namevar (str): Name of the variable to display in next to the legend
    units (str): units of the variable
    axis: in which axis of the bigger figure should be the plot located
    cmap (str) choose the colormap according to the variable, if plotting trends 
        for precipitation is recommendable to use "seismic_r" while 0 correspond to the white color
    """
    
    bound=countrybd.to_crs(data.crs.to_string()) 
    
    gw=gwstat.plot(ax=axis,
               markersize=10, color="darkred",
               marker="v", facecolor="None", alpha=0.2)
    if ((data[column].min()<0) and (data[column].max()>0)):
        cmap="seismic_r"
           
    dat=data.plot(ax=axis,column=column,scheme="headtailbreaks", categorical=False, cmap=cmap, markersize=10,
               marker="v", facecolor="None")
    bound.boundary.plot( ax=axis, alpha=0.5, edgecolor='k', linewidth=1)
    
    #Scalebar
    fig=dat.get_figure()

    if ((data[column].min()<0) and (data[column].max()>0)):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.TwoSlopeNorm(vmin=data[column].min(), 
                                                                      vcenter=0,
                                                                       vmax=data[column].max()))    
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data[column].min(), 
                                                                       vmax=data[column].max()))
    #Colorbar 
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('bottom', size='5%', pad=0.5)
    cbar = fig.colorbar(sm,orientation="horizontal",fraction=0.05,cax=cax)
    cbar.ax.set_xlabel(namevar+" ("+units+")")
#    cbar_step=np.abs(data[column].max()-data[column].min())/3 #Number of ticks to show in the colorbar
    #cbar.ax.set_xticklabels(fontsize=8, weight='bold')
    


    #Scale 
    scalebar = ScaleBar(0.5, "m", dimension="si-length", length_fraction=0.10, location="lower left")
    gw.add_artist(scalebar)
    gw.tick_params(axis='y', which='major', labelsize=8, rotation=90)
    gw.tick_params(axis='x', which='major', labelsize=8, rotation=0)
    startx, endx = gw.get_xlim()
    starty, endy = gw.get_ylim()
    #North arrow
    arrx=endx- endx*0.002
    arry=endy-endy*0.0040
    gw.text(x=arrx-arrx*0.0001, y=arry, s='N', fontsize=16,alpha=0.8)
    gw.arrow(arrx, arry-arry*0.002, 0, 10000, length_includes_head=True,
              head_width=8000, head_length=20000, overhang=.2, ec="k",facecolor='k', alpha=0.4)
    #cx.add_basemap(ax=NSmap, source=cx.providers.Stamen.TonerLabels)
    cx.add_basemap(ax=gw,  crs=data.crs.to_string(), source=cx.providers.Esri.OceanBasemap, alpha=0.5,attribution=False)
                                   
    return axis



        
        
        
        
    
    