import os
import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from styles import style,header


st.set_page_config(
    page_title="برنامج المراقبة المالية المصرفي",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
    )


with st.sidebar:
    st.image("images\\exploration.png", width=150)
    header(st,1,"تصفح البيانات وعرض التفاصيل الخاصة بالجداول بسرعة وسهولة ",font_size="25px",font_weight="bold",text_align="center")
    st.markdown("---")
    
style(st)

COLUMN_NAME = "عنوان العامود رقم : "
@st.cache_data
def get_bankes(): return os.listdir("Database")
@st.cache_data
def get_years(bank_name): return os.listdir("Database/"+bank_name)



col1, col2,col3 = st.columns(3)
with col1:
    bank_name = st.selectbox(options=get_bankes(),label_visibility="collapsed",label=" ")
with col2:
    header(st,1,bank_name,font_size="20px")
with col3:
    header(st,1,"ادارة مصرف",font_size="20px")
    
col1, col2,col3 = st.columns(3)
with col1:
    year = str(st.selectbox(options=get_years(bank_name),label_visibility="collapsed",label=" "))
with col2:
    header(st,1,year,font_size="20px")
with col3:
    header(st,1,"الحسابات للسنة المالية",font_size="20px")
st.markdown("---")

    
    
tables_input =  [i for i in os.listdir("Database/"+bank_name+"/"+year)] 


header(st,1,"القوائم المتوفرة",font_size="20px")
file_name = st.selectbox(label=" ", options=tables_input,label_visibility="collapsed")
header(st,1,"حدد القائمة لاستعراض البيانات ",font_size="20px",text_align="center")
header(st,1,"قم باختيار الملف",font_size="20px")
file_path = st.file_uploader(label=" ",type=["csv","xlsx"],label_visibility="collapsed")
analysis_btn = st.button("تحليل")
st.markdown("---")
header(st,1,f"{file_name.split('.')[0]}",font_size="20px",text_align="center")
st.markdown("---")


if analysis_btn:
    if file_path is not None:
        df = pd.read_excel(file_path)
        df.columns = [i for i in range(len(df.columns))]
        df = df.where(pd.notnull(df), "-")
        st.dataframe(df,use_container_width=True)
        st_profile_report(ProfileReport(df, title=f"{file_name.split('.')[0]}"))
    else:
        path = "Database/"+bank_name+"/"+year+"/"+file_name
        if file_name.endswith("csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)        
        df.columns = [i for i in range(len(df.columns))]
        df = df.where(pd.notnull(df), "")
        st.dataframe(df,use_container_width=True)
        st_profile_report(ProfileReport(df, title=f"{file_name.split('.')[0]}"))

    
