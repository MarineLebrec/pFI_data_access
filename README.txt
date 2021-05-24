This notebook is to be used for reading and analyzing raw programmable Flow Injection (pFI) data from GlobalFIA FloZF Software.

Full description of FloZF software can be found at: https://www.globalfia.com/store/view/productdetails/virtuemart_product_id/151/virtuemart_category_id/12

------------------------------------------------------------------------------------

Requirements for running notebook:

 - Directory of raw FloZF files in master_data directory, as follows: pFI_Analysis/master_data/

 - Import all necessary packages and modules (also embedded in notebook):
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

------------------------------------------------------------------------------------

Smirnov-Grubbs outlier test library installation:
pip install outlier_utils
License: MIT License
Author: Masashi Shibata
More information: https://pypi.org/project/outlier_utils/