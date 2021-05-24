'''A module for analyzing FloZF programmable Flow Injection data'''

import pfi
import pandas as pd
from glob import glob 
import matplotlib.pyplot as plt
import matplotlib.axis as ax
import numpy as np
from scipy import stats
from tabulate import tabulate
from outliers import smirnov_grubbs as grubbs
from numpy.polynomial import Polynomial as P

def plot_spectra(index):
    '''
    INPUTS: Value of sample index used for plotting absorbance spectra
    OUTPUTS: Absorbance spectra plot
    '''
    
    df = pd.read_csv('master_data/sample_'+ str(index) + '.spectrum.pfl', delimiter='\t', skiprows=3) #imports spectrum file
    spectra = df[(df['Wavelength (nm)'] > 500) & (df['Wavelength (nm)'] <1000)] #keeps data from wavelengths only between 500-1000nm.

    #plots spectra
    plt.figure()
    plt.plot(spectra['Wavelength (nm)'],spectra['Absorbance'])
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Absorbance[mAU]')
    plt.title('Index ' + str(index) + ' Absorbance Spectrum')
    plt.ylim(min(spectra['Absorbance'] - 0.01),max(spectra['Absorbance']+0.01))

    plt.show()

#################################################################################    
    
def plot_timeseries(index,reflambda):
    '''
    INPUTS: 
    - Value of sample index used for plotting time-series monitoring
    - Reference wavelength used for plotting time-series (e.g. A880-A510 or A880-A975)
    OUTPUTS: Time-series plot
    '''
    df = pd.read_csv('master_data/sample_'+ str(index) + '.pfl', delimiter='\t', skiprows=31) #imports time-series file 

    #plots time-series
    plt.figure()
    plt.plot(df['Time (s)'],df[str(reflambda)])
    plt.xlabel('Time[s]')
    plt.ylabel('Absorbance[mAU]')
    plt.title('Index ' + str(index) + ' Absorbance Time Series')
    
    plt.show()
    
#################################################################################    
    
def abs_lookup(index):
    '''
    INPUTS: Value of sample index used for idenfiying absorbance value
    OUTPUTS: Absorbance value of given sample index at 880-510nm
    '''
    file_list = glob('master_data/*.pfl') #produces list of all pfl files in master_data directory 

    indeces = [] 
    absorbances = dict()
    A880_A510 = dict()

    for file in file_list: #extracts the index numbers from each file as unique idenfiers and stores them in a list "indeces"
        short_title = file.split('_')[2]
        name = short_title.split('.')
        number = int(name[0]) 
        indeces.append(number)
    
        #Cleaning master_data time-series files to correspond to correct reference wavelength in first column of file 
        absorbances[number] = pd.read_csv(file, delimiter='\t', nrows=30)
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        Abs = absorbances[number]['sample name'] == "A880-A510" #Extracting absorbance values that only correspond to the 510nm reference wavelength.
        A880_A510[number] = (absorbances[number][Abs]['sample']) #Dictionary which contains absorbance values at 880-510nm for any index.
        
    return 'Absorbance[mAU] @ 880-510nm = ' + A880_A510[index]
  
#################################################################################    

def multispectra(sampleindeces):
    
    '''
    INPUTS: Values of sample indeces used for plotting multiple absorbance spectra (1-D array)
    OUTPUTS: Absorbance spectra plot of all specified samples together
    '''
    
    spectra_list = glob('master_data/*.spectrum.pfl')

    indeces = []
    spectra = dict()
    A880 = dict()

    for file in spectra_list:
        short_title = file.split('_')[2]
        name = short_title.split('.')
        number = int(name[0])
        indeces.append(number) #produces list of all pfl files in master_data directory 
        spectra[number] = pd.read_csv(file, delimiter='\t', skiprows=3) #imports all spectra from master_data directory
    
    
   #plots all specified spectra in one figure
    for i in sampleindeces:
        plt.plot(spectra[i]['Wavelength (nm)'],spectra[i]['Absorbance'],label=str(i))
        plt.xlabel('Wavelength [$\lambda$]')
        plt.ylabel('Absorbance [mAU]')
        plt.title('Absorbance Spectra')
        plt.xlim(500,1000)
        plt.ylim(-0.1,0.5)
        plt.legend()
        
    plt.show()

#################################################################################    
    
def calib_plots(indeces):
    '''
    Inputs:
    indeces - List of sample indeces used in calibration. Note: must be same length as the x variable [1-D array]
    Note: assumes calibration uses standards of 0uM, 0.5uM, 1.5uM, 3uM run in triplicates.
    
    Returns: 
    Linear least-squares regression plot + residuals from model
    2nd degree Polynomial least-squares regression plot + residuals from model
    '''
    
    x = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 3, 3, 3])
    
    
    absorbances = dict()
    A880_A510 = dict()
    y = []

    for number in indeces:
        absorbances[number] = pd.read_csv('master_data/sample_'+str(number)+'.pfl', delimiter='\t', nrows=30)
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        Abs = absorbances[number]['sample name'] == "A880-A510" #Extracting absorbance values that only correspond to the 510nm reference wavelength.
        A880_A510[number] = float(absorbances[number][Abs]['sample']) #Completed dictionary which contains absorbance values at 880-510nm for any index.
            
        val = A880_A510[number]
        y.append(val) #list of absorbance values for all specified indeces in input

    
    results = stats.linregress(x,y) #calcualting linear least-squares regression from inputs
    linear = (results[0]*x) + results[1]
    residual = y - linear 
    
    fit = np.polyfit(x,y,2) #calcualting 2nd degree polynomial from inputs

    a = fit[0]
    b = fit[1]
    c = fit[2]
    fit_equation = a * np.square(x) + b * x + c
    
    
    #PLOTTING LINEAR REGRESSION
    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    plt.plot(x,y,'.',ms=10)
    plt.plot(x,linear)
    plt.title('Calibration: Linear Fit')
    plt.xlabel('PO4 [uM]')
    plt.ylabel('Absorbance [mAU]')
    plt.grid()
    
    #RESIDUALS OF LINEAR CALIBRATION
    plt.subplot(2,2,2)
    plt.plot(x, y-linear, '.',ms=10)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)
    plt.ylabel('Residuals')
    plt.xlabel('PO4 [uM]')
    plt.title('Residuals: Linear Fit')
    plt.grid()

    
    #PLOTTING 2ND DEG POLYNOMIAL FIT
    plt.subplot(2,2,3)
    plt.plot(x,y,'.',ms=10)
    plt.plot(x,fit_equation)
    plt.ylabel('Absorbance [mAU]')
    plt.xlabel('PO4 [uM]')
    plt.title('Calibration: Polynomial Fit')
    plt.grid()

    
    #RESIDUALS OF POLYNOMIAL FIT
    plt.subplot(2,2,4)   
    plt.plot(y,y-fit_equation,'.',ms=10)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)
    plt.ylabel('Residuals')
    plt.xlabel('PO4 [uM]')
    plt.title('Residuals: Polynomial Fit')
    plt.grid()

    plt.show()

#################################################################################    

def calib_stats(indeces):
    '''
    Inputs:
    indeces - List of sample indeces used in calibration. Note: must be same length as the x-array (array of known PO4 concentrations used in calibration)
    
    Outputs:
    Table containing figures of merit: calibration slope, calibration y-intercept, coefficient of determination (r-squared), standard deviation of the slope, 95% confidence intervals of the slope (using alpha = 0.05), Limit of Detection (nM). 
    '''
    
    x = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 3, 3, 3])
    
    absorbances = dict()
    A880_A510 = dict()
    y = []

#Cleaning master_data time-series files to correspond to correct reference wavelength in first column of file 
    for number in indeces:
        absorbances[number] = pd.read_csv('master_data/sample_'+str(number)+'.pfl', delimiter='\t', nrows=30)
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        Abs = absorbances[number]['sample name'] == "A880-A510" #Extracting absorbance values that only correspond to the 510nm reference wavelength.
        A880_A510[number] = float(absorbances[number][Abs]['sample']) #Completed dictionary which contains absorbance values at 880-510nm for any index.
            
        abslist = A880_A510[number]
        y.append(abslist) #list of absorbance values for all specified indeces in input

    lin_results = stats.linregress(x,y) #linear least-squares regression fit
    a2 = lin_results[0] #linear slope 
    intercept = lin_results [1] #intercept of linear model
    y_hat = (a2*x) + intercept

    
    poly_results = np.polyfit(x,y,deg=2) #polynomial fit 
    
    r_squared = lin_results.rvalue**2
    
    N=len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    s_yx = np.sqrt(sum((y-y_hat)**2)/(N-2)) #random error in the y-direction

    s_m = s_yx / (np.sqrt(sum((x-mean_x)**2))) #standard deviation of the slope
    
    blanks = y[0:3] #specifies blank standards. Note: assumes that first 3 datapoints in calibration are blanks in triplicate
    
    LOD = 1000 * ((3*np.std(blanks))/a2) #Limit of detection (nm)
    
    ####Calculating 95% Confidence interval of the slope####
    
    variance_xy = sum((x-mean_x)*(y-mean_y))/(N-1) #covariance of x and y
    variance_x = (sum((x-mean_x)**2))/(N-1) #variance of x
    variance_y = (sum((y-mean_y)**2))/(N-1) #variance of y
    
    r_xy = variance_xy / (np.sqrt(variance_x)*np.sqrt(variance_y)) #Calculates Pearson correlation (r)   
    
    alpha=0.05
                           
    t = stats.t.ppf(1-alpha/2,N-2) #calculates critical t-value using alpha and degrees of freedom  
    
    SE_a2 = (sum((y-y_hat)**2) / sum((x-mean_x)**2)) / np.sqrt(N-2) #standard error of the slope
    
    CI = t*SE_a2 #95% Confidence Interval of the Slope
    
    table = [['Slope', 'Intercept', 'R-squared', 'Standard Deviation of the Slope', '95% CI of the Slope','Limit of Detection (nm)'], [a2, intercept, r_squared, s_m, CI, LOD]]
    
    return print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

#################################################################################    

def outlier_test(values):
    '''
    Inputs: Absorbance to be used in outlier test 
    Outputs: If outlier exists, will print "outlier exists" statment + value that is deemed an outlier. If no outlier exists, "No outlier" statement is printed.
    '''
    relstdev = 100 * (np.std(values)/np.mean(values))
    if (relstdev > 10):
        outlier = grubbs.max_test_outliers(values, alpha=.05) 
        return print('Outlier exists:',outlier)
    
    else: print('No outlier')

#################################################################################    

def slope_from_index(indeces):
    '''
    INPUTS: 
    indeces = List of indeces used in a calibration [1-D array]
    OUTPUTS: Slope of linear regression model from calibration.
    '''

    x = [0,0,0,0.5,0.5,0.5,1.5,1.5,1.5,3,3,3]

    absorbances = []
    y = []

    for i in indeces:
        absorbances = pd.read_csv('master_data/sample_'+ str(i) + '.pfl', delimiter='\t', nrows=30)
        
        absorbances = absorbances.replace(to_replace=absorbances['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances = absorbances.replace(to_replace=absorbances['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances = absorbances.replace(to_replace=absorbances['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        
        Abs = absorbances['sample name'] == "A880-A510"
        A880_A510 = float(absorbances[Abs]['sample'])
        y.append(A880_A510)
        
    lin_results = stats.linregress(x,y)
    slope = lin_results[0]
    
    return slope

#################################################################################    

def solve_for_conc(calib_indeces, sample_indeces):
    '''INPUTS: 
    Phosphate concentrations of calibration [1D array]
    Absorbance values (mAU) [1D array]
    
    OUTPUTS:
    Concentration values corresponding to absorbance readings, using 2-degree polynomial calibration curve [1D array)]
    '''
    x = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 3, 3, 3]) #concentrations of standards used in calibration (in triplicates)
    
    absorbances = dict()
    A880_A510 = dict()
    calib_y = []

    
#Extracting list of absorbance values from calibration curve, using x and calib_indeces inputs. 
    for number in calib_indeces:
        absorbances[number] = pd.read_csv('master_data/sample_'+str(number)+'.pfl', delimiter='\t', nrows=30)
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        Abs = absorbances[number]['sample name'] == "A880-A510" #Extracting absorbance values that only correspond to the 510nm reference wavelength.
        A880_A510[number] = float(absorbances[number][Abs]['sample']) #Completed dictionary which contains absorbance values at 880-510nm for any index.
            
        abslist = A880_A510[number]
        calib_y.append(abslist) 
        
        
#Extracting list of absorbance values from measured sample with unknown PO4 concentration, using sample_indeces inputs.     
    absorbances = dict()
    A880_A510 = dict()
    sample_y = []

    for number in sample_indeces:
        absorbances[number] = pd.read_csv('master_data/sample_'+str(number)+'.pfl', delimiter='\t', nrows=30)
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances[number] = absorbances[number].replace(to_replace=absorbances[number]['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        Abs = absorbances[number]['sample name'] == "A880-A510" #Extracting absorbance values that only correspond to the 510nm reference wavelength.
        A880_A510[number] = float(absorbances[number][Abs]['sample']) #Completed dictionary which contains absorbance values at 880-510nm for any index.
            
        abslist = A880_A510[number]
        sample_y.append(abslist)


    p = P.fit(x, calib_y, 2)
    
    conc_list = []
    for value in sample_y:
        conc = (p - value).roots()[1] #calculates the roots of the polynomial model - for this application, we are only interested in the 2nd root. 
        conc_list.append(conc)
        
    return conc_list

#################################################################################    


if __name__ == '__main__':
	print('Run pFi functions if run as a script')