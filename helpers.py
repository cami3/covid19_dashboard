import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
 
import datatable as dt
from datetime import date, timedelta


@st.cache # Effortless caching
def load_data(DATA_URL, DATE_COLUMN):
    # add argument nrows=nrows if needed to upload only a chunk of data
    data = dt.fread(DATA_URL).to_pandas()
    columnslist = [
    'excess_mortality', 
    'extreme_poverty',
    'female_smokers',
    'hosp_patients',
    'hosp_patients_per_million',
    'icu_patients',
    'icu_patients_per_million',
    'male_smokers',
    'new_tests',
    'new_tests_per_thousand',
    'new_tests_smoothed',
    'new_tests_smoothed_per_thousand',
    'positive_rate',
    'tests_per_case',
    'total_tests',
    'total_tests_per_thousand',
    'weekly_hosp_admissions',
    'weekly_hosp_admissions_per_million',
    'weekly_icu_admissions',
    'weekly_icu_admissions_per_million']
    # importing columns in list as type float
    for col in columnslist: 
        data[col] = data[col].astype(np.float64)
    # specify datetime type for DATE_COLUMN
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], infer_datetime_format=True)
    # Do not sort values
    data.sort_values(by=[DATE_COLUMN], ascending=False, inplace=True)
    data.set_index('date', inplace=True)
    # removing "Our world in data"-added rows with iso_code starting with "OWID"
    data = data[~data['iso_code'].astype(str).str.startswith('OWID')]
    return data


@st.cache(allow_output_mutation=True)
def load_data_ita(DATA_REGIONI_URL, DATA_PROVINCE_URL, DATE_COLUMN):
    # add argument nrows=nrows if needed to upload only a chunk of data
    data_regioni = dt.fread(DATA_REGIONI_URL).to_pandas()
    data_province = dt.fread(DATA_PROVINCE_URL, fill=True).to_pandas()
    data_regioni[DATE_COLUMN] = pd.to_datetime(data_regioni[DATE_COLUMN], infer_datetime_format=True)
    data_province[DATE_COLUMN] = pd.to_datetime(data_province[DATE_COLUMN], infer_datetime_format=True)
    data_regioni.sort_values(by=[DATE_COLUMN], ascending=False, inplace=True)
    data_province.sort_values(by=[DATE_COLUMN], ascending=False, inplace=True)
    data_regioni.set_index('data', inplace=True)
    data_province.set_index('data', inplace=True)
    # removing "Our world in data"-added rows with iso_code starting with "OWID"
    return data_regioni, data_province

def rollup_1day(datadff, column):
    #latest_column = datadff.rolling('-730D', on=datadff.index)[column].max() ## non funziona!!
    latest_column = datadff[column].bfill()
    
    dataframe = datadff.merge(latest_column, left_index=True, right_index=True)
    dataframe['%s' %(column)] = dataframe['%s_x' %(column)]
    dataframe = dataframe.drop(('%s_x' %(column)), axis=1)
    dataframe[('latest_%s' %(column))] = dataframe[('%s_y' %(column))]
    dataframe = dataframe.drop(('%s_y' %(column)), axis=1)
    
    return dataframe




def shift_1day(datadfff, column):
    #shift_col = datadfff.shift(periods=1, freq='D')[[('%s'%(column)),'continent', 'location']]
    shift_cols = datadfff.shift(periods=1, freq='D', axis='index')[('%s' %(column))]
    shift_cols.rename(('%s_on_previous_day' %(column)), inplace=True)
    #st.write(shift_cols)
    #st.write(datadfff[[('%s'%(column)),'continent', 'location', 'iso_code']])
    dataframe = datadfff.join(shift_cols, how='left')
    #st.write(dataframe[['total_deaths', 'total_deaths_on_previous_day']])
    #dataframe[('%s_on_previous_day'%(column))] = dataframe[('%s_y'%(column))]
    #dataframe[('%s'%(column))] = dataframe[('%s_x'%(column))]
    #dataframe = dataframe.drop(columns=[('%s_y'%(column)), (('%s_x'%(column)))])
    return dataframe






