# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:50:29 2018

@author: 한승표
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dateutil.relativedelta import relativedelta
raw_data = pd.read_excel(r'data_과제1.xlsx', sheet_name = 'Sheet2',index_col = 0,index_name=['Mat'])
raw_data.index = raw_data.index.date

sigma = 0.003
reference_day =datetime.date(2018,10,29)
def libor_structure(raw_data,reference_day):
    raw_data['DF'] = np.zeros(len(raw_data))
    forward_rate=np.zeros(len(raw_data))
    for i in range(len(raw_data)):
        t_delta = raw_data.index[i]-raw_data.index[1]
        if raw_data['Instrument'][i] =='MMD':
            Z_ts = 1/((1+(1/  360)*(raw_data.iloc[0,2]/100))*(1+(1/360)*(raw_data.iloc[1,2]/100))) 
            raw_data['DF'][i] = Z_ts/(1+((t_delta.days)/360)*(raw_data.iloc[i,2]/100))
            
        if raw_data['Instrument'][i] =='Futures':
            if raw_data.index[i].month == raw_data.index[i-1].month:
                raw_data['DF'][i] = raw_data['DF'][i-1]
            T1 = (raw_data.index[i] - reference_day).days / 360
            T2 = (raw_data.index[i]+datetime.timedelta(days=30) -reference_day).days /360
            forward_rate[i]=(((100-raw_data.iloc[i,2])/100) - (sigma**2 *T1 *T2 /2)) 
            raw_data['DF'][i+1] =   raw_data['DF'][i] / (1+(datetime.timedelta(days=30).days / 360)*forward_rate[i])
        
        if raw_data['Instrument'][i] =='Swap':
            H = raw_data.iloc[i,2]
            delta = 0.5
            num = np.array(2*((raw_data.index[i] - reference_day).days / 360)).round()
            df = np.zeros(int(num))
            index = []
            for j in np.arange(np.array(2*((raw_data.index[-1] - reference_day).days / 360)).round()+1):
                index.append(reference_day + relativedelta(months=6*j))
            
            swap_data = pd.DataFrame(columns=['DF'],index = index)[1:]
    
            for j in np.arange(1,int(num)):
                date = reference_day + relativedelta(months=6*j)
                if date in raw_data.index:
                    swap_data.loc[date] = raw_data.loc[date]
                else:
                    swap_data.loc[date] = swap_data.loc[date-relativedelta(months=6)]
                df[j-1] = (H/100) * delta * swap_data.iloc[j-1,0]
            raw_data.iloc[i,3] = (1- sum(df))/(1+(H/100)*delta)
            
    libor = pd.DataFrame(raw_data['DF'])
    libor['MAT'] = np.zeros(len(libor))
    libor['libor'] = np.zeros(len(libor))
    for i in range(len(libor)):
        libor['MAT'][i] = ((libor.index[i] -reference_day).days/360)
        libor['libor'][i] = -np.log(libor['DF'][i])/libor['MAT'][i]
    return libor
#%%
#interpolatiion
libor = libor_structure(raw_data,reference_day)[1:]
libor['MAT'][-1] =10
maturity = libor['MAT'].values
spot_rate = libor['libor'].values*100
ff = interp1d(maturity, spot_rate,kind='zero')
x = maturity
y = spot_rate
xnew = np.arange(0.01,10.01,0.01)
ynew = ff(xnew)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, y,'rs', label='LIBOR Rates')
ax.plot(xnew, ynew,'b-', label='Interpolated LIBOR Curve')
plt.xlabel('Maturity')
plt.ylabel('LIBOR')
plt.title('USD LIBOR curve')
ax.legend()
plt.show()
#plt.cla()
