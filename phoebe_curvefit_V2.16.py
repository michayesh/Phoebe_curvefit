#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:13:36 2018
Phoebe model loop based on RV sup_conj
may 2018
Version 26/8/18 

Version 29/8/18
Version 1.1 with T1,M1,R1 as params 3/9/18

Version 2.00:
  14/11/18  with constrain of main seq secondary
  14/11/18  with M1,R1,T1 added to the chi2
  21/11/18  multithread
  
Version 2.10:
  26/11/18 removed writelog
version 2.12:
    number of threads is determined by param
Version 2.14:
  the bundle name k in the main is replaced by kb
Version 2.15:
  18/12/18
  support mpi run (on phoebe ver 2.1.1)
  Version 2.16:
  19/12/18
  automatically retrieve input files

@author: Micha
"""
import sys
import os
import time
import phoebe
import pickle
from glob import glob
import json
#from phoebe import u # units
import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use('micha')
#import scipy.optimize as op
#import matplotlib.image as mpimg
from datetime import datetime


allStart=time.time()
#
#% setup  general parameters 
#logg_sun=4.43812 
KeplerOffset=2454833.0 # times relative to this offset will be prefixed by k

# number of threads to run simple fit
# if n_threads == 0 run simple fit without threading
n_threads = 0

machine = os.uname()[1]
#machine = 'engelmic@astrophys.tau.ac.il'
if machine == 'user-Latitude-E5570':
    rootpath = r'/home/user/Dropbox/KBEER_phoebe_ipynb/'
    sys.path.append('/home/user/Dropbox/spyder/') # path for pyBEER
    location='dell'
    if n_threads==0:
        phoebe.mpi_on(nprocs=4) # 
        
    
elif machine == 'eshel-blue':
    rootpath = r'/home/eshel-blue/micha/dropboxClone/'
    sys.path.append('/home/eshel-blue/micha/dropboxClone/spyder') # path for pyBEER
    location = 'blue'
    if n_threads==0:
        phoebe.mpi_on(nprocs=6) # 
    
elif machine == 'engelmic@astrophys.tau.ac.il':
     rootpath = r'/storage/home/engelmic/KBEER/'
     sys.path.append('/storage/home/engelmic/scripts') # path for pyBEER
     location = 'astro'
     
else:
    print('unknown Machine - setup paths.')
    sys.exit()
import pyBEERnas as pb
import simpleFit as sf     



        
#%% Initialize the variables
print('Phoebe Version: %s'% phoebe.__version__)
kb= phoebe.default_binary()
kb.flip_constraint('mass@primary', 'sma')

#%%
#
sysNum = 3


# Choose the System
############################# sysNum=1 ########################################
if sysNum == 1:
    sysName = 'KIC04931390'
 
    dataPath = rootpath +'KIC04931390/'
    configPath = rootpath +'KIC04931390/currentConfig/'
    # starting point
    # config file gives the "envelope params"
#    configFileName='KIC05200778_20181217_start.json'
#    #initial point determined by fitpar init
#    fitParamsFileName='KIC05200778_20181217_fitpar_start.json'
    
# LC file name - text file containig times and Dfs (ppm) with zero mean
    fullLCfileName = 'cf_KIC04931390_LC.dat'
#    
    #unfolded RV data with zero mean and errors scaled to >1.5 km/s
    RVDatafn=  'cf_KIC04931390_RVscaled.dat'
#    
#   chosenParams= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #all
#    chosenParams= [0,1] # P t0   
#    chosenParams= [4,5,6,7] #incl,q,T2,R2
#    ["P", "t0s", "e", "w", "incl", "T1" M1", R1,"q", "T2", "R2", 
#"rho1", "rho2", "gb1", "gb2","ldc11","ldc12","ldc21","ldc22"]
#    chosenParams= [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    chosenParams= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] # all

#%%########################## sysNum=2  ########################################
# KIC06292925
if sysNum == 2:
    sysName = 'KIC06292925'
 
    dataPath = rootpath +'KIC06292925/'
    configPath = rootpath +'KIC06292925/currentConfig/'
    # starting point
    # config file gives the "envelope params"
#    configFileName='KIC06292925_20181127_config.json'
#    #initial point determined by fitpar init
#    fitParamsFileName=''
    
#    
    fullLCfileName = 'cf_KIC06292925_LC.dat'
#    
    #unfolded RV data
    RVDatafn=  'cf_KIC06292925_scaledRV.dat'
#    binnedLCdatafn  = 'KIC06292925_c_binned.dat'
#    ["P", "t0s", "e", "w", "incl", "T1" M1", R1,"q", "T2", "R2", 
#"rho1", "rho2", "gb1", "gb2","ldc11","ldc12","ldc21","ldc22"]
#    chosenParams= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #all
    chosenParams= [2,3,4,5,6,7,8,11,12] # exclude P t0s T2,R2 and "gb1", "gb2","ldc11","ldc12","ldc21","ldc22"
#    chosenParams= [0,1] # P t0
#%%############################ sysNum=3 #########################################
## KIC05200778
if sysNum == 3:
    sysName = 'KIC05200778'
    dataPath  =  rootpath +'KIC05200778/'
    configPath = rootpath +'KIC05200778/currentConfig/'
    # starting point
    # config file gives the "envelope params"
#    configFileName='KIC05200778_20181219_0951_config_start.json'
#    #initial point determined by fitpar init
#    fitParamsFileName='KIC05200778_20181219_0951_fitpar_start.json'
    # LC file name - text file containig times and Dfs (ppm) with zero mean
    fullLCfileName = 'cf_KIC05200778_LC.dat'
    #unfolded RV data with zero mean and errors scaled to >1.5 km/s
    RVDatafn=  'cf_KIC05200778_scaledRV.dat'
#    chosenParams= [0,1] # P t0   
#    chosenParams= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #all 
    chosenParams= [2,3,4,5,6,7,8,11,12] # exclude P t0s T2,R2 and "gb1", "gb2","ldc11","ldc12","ldc21","ldc22"
    



#%%
#if wrt2log: fd = open(outputPath+logFileName,'w')


#foldedRVDatafn= sysName + 'foldedRV.dat'


modelPhasedLCdatafn =sysName + '_c_modelPhasedLC.dat'
# TODO print the script name
#print('phoebe_curvefit_V1.111.py')
#if wrt2log:fd.write('phoebe_curvefit_V1.111.py')
print('system is: %s' % sysName)
#if wrt2log:fd.write('system is: %s \n' % sysName)

#if wrt2log: fd.write('output will be written to %s \n'% outputPath)

fitParamsFileName = glob(configPath+'*_fitpar_start.json')[0]
configFileName = glob(configPath+'*_config_start.json')[0]



print('The files used:')
#if wrt2log: fd.write('The files used:')
print(configFileName+'  '+fitParamsFileName+'  '+fullLCfileName + ' ' + RVDatafn)
#if wrt2log: fd.write(configFileName+'  '+fitParamsFileName+'  '+fullLCfileName)
print('Chosen params:'+str(chosenParams))
#if wrt2log: fd.write('Chosen params:'+str(chosenParams))
#if wrt2log: fd.flush()
#if raw_input('ok?') != 'y':
#    sys.exit()

#

sysConfig = json.load(open(configFileName))

fitParams = json.load(open(fitParamsFileName))



#%%######################create output folder #################################
# 
outputPath = dataPath + datetime.now().strftime('%Y%m%d_%H%M')+'_'+location+'/'
print('output will be written to %s'% outputPath)
logFileName = sysName + '_' +datetime.now().strftime('%Y%m%d_%H%M')+'_log.txt'
os.mkdir(outputPath,)

tl=datetime.now().strftime('%Y%m%d_%H%M')
loggerFileName=outputPath+sysName +'_'+tl+'.log'
logger = phoebe.logger(clevel='CRITICAL', flevel='DEBUG', filename=loggerFileName)

print( sysConfig)
#if wrt2log: fd.write('Sys. Config.:\n')
#if wrt2log: fd.write(str(sysConfig)+'\n')
#if wrt2log: fd.flush()
#%%  set the parameters into the bundle

#k['period@binary@component']=sysConfig['period']
kb=pb.sysConfig2Bundle(sysConfig,kb)
period=sysConfig['period']


sysfm = sysConfig['measMassfn']
initM1 = sysConfig['M1']
initR1 = sysConfig['rp1']
initT1 = sysConfig['teff1']
#%% Read full LC

fullLCtimes, fullLCDfs = np.loadtxt(configPath + fullLCfileName,
                                       delimiter=',',unpack=True )
fullLCDfs -=  fullLCDfs.mean() # set zero mean

#%%
#  Read unfolded RV data (zero mean at measure dates)
measRVtimes, measRVs, measRVsigmas = np.loadtxt(configPath + RVDatafn, delimiter=',',unpack=True)
# 

# check that the mean is 0
measRVs -= measRVs.mean()
print('Mean of Meas. RV data set= %.5g '% measRVs.mean())
#if wrt2log: fd.write('Mean of Meas. RV data set= %.5g '% measRVs.mean())
#%%
# ########## Generate RV data set for phoebe
## RV data set
if 'RV_Meas' in kb.datasets:
    kb.remove_dataset('RV_Meas')
kb.add_dataset('rv', times=measRVtimes, rvs=measRVs, sigmas=measRVsigmas, dataset='RV_Meas')

kb['passband@RV_Meas@dataset']='Kepler:mean'

    

#print('The datasets to be computed are: \n {}'.format(k.datasets))
#if wrt2log: fd.write('The datasets to be computed are: \n {}'.format(k.datasets))
#if wrt2log: fd.flush()   
    #%%  set number of triangles

ntr1=10000 # number of triangles prim.
ntr2=10000 ## number of triangles sec.

#ntr1=1000 # number of triangles prim.
#ntr2=1000 ## number of triangles sec.
kb.set_value('ntriangles@primary', ntr1)
kb.set_value('ntriangles@secondary', ntr2)
print('ntr1 = %7g   ntr2 = %7g'% (kb['ntriangles@primary'].value,kb['ntriangles@secondary'].value))
#if wrt2log: fd.write('ntr1 = %7g   ntr2 = %7g'% (k['ntriangles@primary'].value,k['ntriangles@secondary'].value))
#if wrt2log: fd.flush()   

#%%

#def make_calcLC(k_in,fd):
def make_calcLC(k_in):
#def make_calcLC(k):
#    @Memoize
    def calcLC(times,*pars):
      
      # duplicate
        k=phoebe.Bundle(k_in.to_json())
        
        spars=pars*scale + bias
        spars_list=spars.tolist()
        # load the params to the bundle
        pb.loadParams2Bundle(k,names,spars_list) 
        print('Params loaded.')
        # constrain MS secondary
        curM2=k['mass@primary@star@component'].value*k['q@binary@component'].value
        [msR2,msT2,curlogg] =pb.getMS_RadTemp(curM2)
        curT2 = max([msT2,4000.0])
        k.set_value('teff@secondary@star@component', curT2)
        k.set_value('requiv@secondary@star@component', msR2)
        
        
        print('\n'+ datetime.now().strftime('%d/%m/%y %H:%M:%S')+'\n')
#        if wrt2log: fd.write('\n'+ datetime.now().strftime('%d/%m/%y %H:%M')+'\n')
        print(zip(names,spars.tolist()))
#        if wrt2log: fd.write(str(zip(names,spars.tolist())))
        
        history['param'].append(spars)
        history['sec'].append([curM2,msR2,curT2])
        
        print('M2= %8g R2 = %8g T2 = %8g'% (curM2,msR2, curT2))
#        if wrt2log: fd.write('M2= %8g R2 = %8g T2 = %8g'% (curM2,msR2, curT2))
        # save current bundle
        bundle_file_name= 'current_bundle.phoebe'
#        bundle_path = 
        k.save(outputPath+bundle_file_name)
        print('Saved curent bundle.')
        print('calculating...')
#        if wrt2log: fd.write('\n calculating...')
#        if wrt2log: fd.flush()
        
        # here we have the current t0 and P 
        curP = k['period@binary@component'].value
        curt0s = k['t0_supconj@binary@component'].value
        phases= np.arange(0.,1.01,0.01)+ 0.005
#       calcTimes = curt0s + curP * np.arange(0.,1.01,0.01)+0.005
        calcTimes = curt0s + curP * phases
        
        ########## Generate LC data set for phoebe
        ## 
        if 'LCcalc' in k.datasets:
            k.remove_dataset('LCcalc')
            # k.add_dataset('lc', times=calcTimes, fluxes=measDfs, sigmas=measDfsigmas, dataset='LCcalc')
        k.add_dataset('lc', times=calcTimes, dataset='LCcalc')
        k['passband@LCcalc@dataset']='Kepler:mean'
        
#        print('The datasets to be computed are: \n {}'.format(k.datasets))
        
        k.run_compute(model='curModel')
#        fluxes = k['fluxes@curModel@bLCMeas'].value
        fluxes = k['fluxes@curModel@LCcalc'].value
#        times = k['times@curModel@LCcalc'].value
        Dfs=pb.flux2DF(fluxes)*1.0e6
#        print('Dfs mean= %.4g'% Dfs.mean())
#        phases = ((times-curt0s) % curP)/curP #
       
        edgevalue=np.mean([Dfs[0],Dfs[-1]])
        phases = np.append(phases,1.0)
        phases = np.insert(phases,0,0.0)
        Dfs = np.append(Dfs,edgevalue)
        Dfs = np.insert(Dfs,0,edgevalue)
        Dfs -= Dfs.mean()
        fullLCphases = ((fullLCtimes-curt0s) % curP)/curP
             
#        
        fullModelDfs =np.interp(fullLCphases,phases,Dfs)
        fullModelDfs -= fullModelDfs.mean()
#        print('fullmodelDfs mean = %.5g'% fullModelDfs.mean())
        
        RVs = k['rvs@primary@curModel@RV_Meas'].value
        RVs -= RVs.mean()
#        print('modelRVs mean= %.5g'% RVs.mean())
        
        curM1 = k['mass@primary@star@component'].value
        curR1 = k['requiv@primary@star@component'].value
        curT1 = k['teff@primary@star@component'].value
        chi2fullLC = pb.chisq(fullLCDfs,fullModelDfs,full_scaled_measDfsigmas)
        history['full'].append(chi2fullLC)
#        chi2LC = pb.chisq(measDfs,Dfs,measDfsigmas)
#        chi2LCnorm = pb.chisq(measDfs,Dfs,scaled_measDfsigmas)
        chi2RV = pb.chisq(measRVs,RVs,measRVsigmas)
        history['RV'].append(chi2RV)
        chi2RVnorm = pb.chisq(measRVs,RVs,scaled_measRVsigmas)
        history['RVnorm'].append(chi2RVnorm)
        print('chi2fullLC = %8g \n'%(chi2fullLC))
#        if wrt2log: fd.write('chi2fullLC = %8g \n'%(chi2fullLC))
        print('chi2RV= %8g (chi2RVnorm= %8g) \n'%(chi2RV,chi2RVnorm))
#        if wrt2log: fd.write('chi2RV= %8g (chi2RVnorm= %8g) \n'%(chi2RV,chi2RVnorm))
        
#        if wrt2log: fd.flush()
        history_file_name = outputPath+sysName +'_'+'current_history.pkl'
        pickle.dump(history,open(history_file_name,'w'))
        print('Saved current history')
        curMRT1 =np.array([curM1,curR1,curT1])
        model = np.concatenate((fullModelDfs,RVs,curMRT1))
        return(model)
        
    return calcLC
    
#%%  define the initial point  



scale = np.asarray([fitParams['scale'][p] for p in chosenParams])

bias = np.asarray([fitParams['bias'][p] for p in chosenParams])

initPars = (np.asarray([fitParams['init'][p] for p in chosenParams])-bias)/scale



bnds =((np.asarray([fitParams['lbnds'][p] for p in chosenParams])-bias)/scale, 
       (np.asarray([fitParams['ubnds'][p] for p in chosenParams])-bias)/scale)

searchStep = np.asarray([fitParams['step'][p] for p in chosenParams])/scale

names = [fitParams['names'][p] for p in chosenParams]

#scaled_measDfsigmas=measDfsigmas
#scaled_measDfsigmas=10*np.ones(len(measDfsigmas))
scaled_measRVsigmas = measRVsigmas
#scaled_measRVsigmas = 15.0*np.ones(len(measRVsigmas))
#fullLCsigmas = 800*np.ones(len(fullLCtimes))
fullLCsigmas = 400*np.ones(len(fullLCtimes))
full_scaled_measDfsigmas = fullLCsigmas
dummy =np.array([800.,800.0,800.0])
measTimes = np.concatenate((fullLCtimes,measRVtimes,dummy))
#  replace with int values from params
#initM1 = 2.1
#initR1 = 1.8
#initT1 = 8380.0
MRT1=np.array([initM1,initR1,initT1])
measData = np.concatenate((fullLCDfs,measRVs,MRT1))
# print the mean of data chck =0
#print('mean of meas Data (LC&RV): %.5g'% measData.mean())
#sigmaM1 = 0.1 sigmaR1 = 0.1 sigmaT1 = 100.0
sigmaMRT1 =np.array([0.1,0.1,100.0])
measSigmas = np.concatenate((full_scaled_measDfsigmas,scaled_measRVsigmas,sigmaMRT1))


#%%

print('Initial parameters:')
#if wrt2log: fd.write('\n Initial parameters:\n')
print(zip(names,initPars*scale+bias))
#if wrt2log: fd.write(str(zip(names,initPars*scale+bias)))
#if wrt2log: fd.flush()


#pr
print('Param. steps:')
#if wrt2log: fd.write('\n Param. steps: \n')
print(zip(names,searchStep*scale))
#if wrt2log: fd.write(str(zip(names,searchStep*scale))+'\n')
#if wrt2log: fd.flush()
# initialize history lists
#chi2fullLC_history = []
#chi2RV_history =[]
#chi2RVnorm_history = []
#param_history =[]
#history={'full':chi2fullLC_history,'RV':chi2RV_history ,'RVnorm':chi2RVnorm_history,'param':param_history}
history={'full':[],'RV':[] ,'RVnorm':[],'param':[],'sec':[]}



#%%


computeStart=time.time()


curxtol=1.e-8
curftol=1.e-8
curgtol=1.e-8
print('\n ftol = %4g  xtol=  %4g  gtol= %4g \n'% (curftol,curxtol,curgtol))

#if wrt2log: fd.write('\n ftol = %4g  xtol=  %4g  gtol= %4g \n' % (curftol,curxtol,curgtol))
#if wrt2log: fd.flush()
#if  not wrt2log: fd = None

print('phoebe.mpi.enabled %r' % phoebe.mpi.enabled)
print('nprocs= %d' % phoebe.mpi.nprocs)



if n_threads == 0 :
    print('Multi thread disabled')
    optimalPars = sf.simpleCurveFit(make_calcLC(kb),measTimes,measData,p0=initPars,sigma=measSigmas,bounds=bnds,
             verbose=2,diff_step=searchStep,ftol=curftol,xtol=curxtol,gtol=curgtol)

else:
    print('number of threads = %d'% n_threads)
    optimalPars = sf.simpleCurveFitMultiThread(make_calcLC(kb),measTimes,measData,p0=initPars,sigma=measSigmas,bounds=bnds,
             verbose=2,diff_step=searchStep,ftol=curftol,xtol=curxtol,gtol=curgtol,threads=n_threads)


computeEnd=time.time()
print('Completed- Computation time: {0:.1f} s'.format(computeEnd-computeStart))
#if wrt2log: fd.write('Completed- Computation time: {0:.1f} s'.format(computeEnd-computeStart))
#if wrt2log: fd.close()
tl=datetime.now().strftime('%Y%m%d_%H%M')

#%% Get the optimal model
#lastPars=param_history[-1]
# if not exist optimalPars
chi2tot=np.asarray(history['full']) + np.asarray(history['RVnorm'])
optindx=np.argmin(chi2tot)
optSpars=history['param'][optindx]
#optSpars=optimalPars[0]*scale + bias
opt_spars_list = optSpars.tolist()
# function
#%%
pb.loadParams2Bundle(kb,names,opt_spars_list)

    
kb.save(outputPath+sysName+'_'+tl+'opt.phoebe')

# save configfile
optsysConfig = pb.bundle2sysConfig(kb)
optsysConfig['measMassfn'] = sysfm
optConfigFileName=outputPath+sysName +'_'+tl+'_opt.json'
json.dump(optsysConfig,open(optConfigFileName,'w'))

# save history
history_file_name = outputPath+sysName +'_'+tl+ '_history.pkl'
pickle.dump(history,open(history_file_name,'w'))

# save the output of curve fit (params and covariance matrix)
#param2save={'opt':optimalPars[0].tolist(),'cov':optimalPars[1].tolist()}
# with simple fit
#param2save={'opt':optimalPars[0].tolist()}
#
#optParFileName = outputPath+sysName +'_'+tl+'_optPar.json'
#json.dump(param2save,open(optParFileName,'w'))


# plug the optimal pars to the init key
fitParams['names']=names
fitParams['init']=optSpars.tolist()
json.dump(fitParams,open(outputPath+sysName+'_'+tl+'opt_fitpar.json','w'))


########  compute and plot the optimal model























