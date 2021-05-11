'''A module for analyzing FloZF data'''

def plot_spectra(index):
    '''
    INPUTS: value of sample index used for plotting absorbance spectra
    OUTPUTS: Absorbance spectra plot
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('master_data/sample_'+ str(index) + '.spectrum.pfl', delimiter='\t', skiprows=3)
    spectra = df[(df['Wavelength (nm)'] > 500) & (df['Wavelength (nm)'] <1000)] #keeps data from wavelengths only between 500-1000nm.

    plt.figure()
    plt.plot(spectra['Wavelength (nm)'],spectra['Absorbance'])
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Absorbance[mAU]')
    plt.title('Index ' + str(index) + ' Absorbance Spectrum')
    plt.ylim(min(spectra['Absorbance'] - 0.01),max(spectra['Absorbance']+0.01))

    plt.show()

def plot_timeseries(index,reflambda):
    '''
    INPUTS: 
    Value of sample index used for plotting time-series monitoring
    Reference wavelength used for plotting time-series (e.g. A880-A510 or A880-A975)
    OUTPUTS: Time-series plot
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('master_data/sample_'+ str(index) + '.pfl', delimiter='\t', skiprows=31)

    plt.figure()
    plt.plot(df['Time (s)'],df[str(reflambda)])
    plt.xlabel('Time[s]')
    plt.ylabel('Absorbance[mAU]')
    plt.title('Index ' + str(index) + ' Absorbance Time Series')

def calib_plots(x,y):
    '''
    Inputs:
    x - Concentrations of standards 
    y - Absorbance values from calibration run
    
    Returns: 
    Linear least-squares regression plot + residuals from model
    2nd degree Polynomial least-squares regression plot + residuals from model
    '''
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    
    results = stats.linregress(x,y)
    linear = (results[0]*x) + results[1]
    residual = y - linear 
    
    fit = np.polyfit(x,y,2)

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
    plt.ylabel('Absorbance')
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
    plt.ylabel('Absorbance')
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
    plt.savefig('Residuals_plot_nitrite')

    plt.show()


def calib_stats(x,y,alpha=0.5):
    
    from scipy import stats
    from tabulate import tabulate
    import numpy as np


    lin_results = stats.linregress(x,y)
    a2 = lin_results[0] #linear slope 
    intercept = lin_results [1] #intercept of linear model
    y_hat = (a2*x) + intercept

    
    poly_results = np.polyfit(x,y,deg=2)
    
    r_squared = lin_results.rvalue**2
    
    N=len(x)
    mean_x = np.mean(x)
    s_yx = np.sqrt(sum((y-y_hat)**2)/(N-2)) #random error in the y-direction

    s_m = s_yx / (np.sqrt(sum((x-mean_x)**2))) #standard deviation of the slope
    
    blanks = y[0:3] #assumes that first 3 datapoints in calibration are blanks in triplicate
    
    LOD = 1000 * ((3*np.std(blanks))/a2) #Limit of detection (nm)
    
    table = [['Slope', 'Intercept', 'R-squared', 'Standard Deviation of the Slope', 'Limit of Detection (nm)'], [a2, intercept, r_squared, s_m, LOD]]

    
    return print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def slope_from_index(x, indexes):
    '''
    INPUTS: 
    x: 1-D array of standard concentrations used in calibration.
    indexes = List of indexes used in a calibration [1-D array]
    OUTPUTS: Slope of linear regression model from calibration.
    '''
    import pandas as pd
    from scipy import stats

    absorbances = []
    abs_list = []

    
    for i in indexes:
        absorbances = pd.read_csv('master_data/sample_'+ str(i) + '.pfl', delimiter='\t', nrows=30)
        
        absorbances = absorbances.replace(to_replace=absorbances['sample name'][23], value='A880-A510', inplace=False, limit=None, regex=False, method='pad')
        absorbances = absorbances.replace(to_replace=absorbances['sample name'][16], value='A880-A775', inplace=False, limit=None, regex=False, method='pad')
        absorbances = absorbances.replace(to_replace=absorbances['sample name'][9], value='A880-A975', inplace=False, limit=None, regex=False, method='pad')

        
        Abs = absorbances['sample name'] == "A880-A510"
        A880 = float(absorbances[Abs]['sample'])
        abs_list.append(A880)
        
    lin_results = stats.linregress(x,abs_list)
    slope = lin_results[0]
    
    return slope



if __name__ == '__main__':
	print('Run pFi functions if run as a script')