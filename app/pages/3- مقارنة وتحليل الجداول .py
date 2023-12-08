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
    st.image("images\\compare.png", width=150)
    header(st,1,"مقارنة القوائم المالية وعرض التفاصيل حسب البيانات المحددة بشكل سلسل",font_size="25px",font_weight="bold",text_align="center")
    st.markdown("---")
    
style(st)
@st.cache_data
def get_bankes(): return os.listdir("Database")
@st.cache_data
def get_years(bank_name): return os.listdir("Database/"+bank_name)



def compare(num):
    files_path = []
    for bank_num in range(num):
        col1, col2,col3 = st.columns(3)
        with col1:
            bank_name = st.selectbox(options=get_bankes(),label_visibility="collapsed",label=" ",key=f"b{bank_num}")
        with col2:
            header(st,1,bank_name,font_size="20px")
        with col3:
            header(st,1,"ادارة مصرف",font_size="20px")
            
        col1, col2,col3 = st.columns(3)
        with col1:
            year = str(st.selectbox(options=get_years(bank_name),label_visibility="collapsed",label=" ",key=f"y{bank_num}"))
        with col2:
            header(st,1,year,font_size="20px")
        with col3:
            header(st,1,"الحسابات للسنة المالية",font_size="20px")
        if num == 1:
            col1, col2,col3 = st.columns(3)
            with col1:
                year2 = str(st.selectbox(options=get_years(bank_name),label_visibility="collapsed",label=" ",key=f"y2{bank_num}"))
            with col2:
                header(st,1,year,font_size="20px")
            with col3:
                header(st,1,"الحسابات للسنة المالية",font_size="20px")

        tables_input =  [i for i in os.listdir("Database/"+bank_name+"/"+year)] 
        header(st,1,"القوائم المتوفرة",font_size="20px")
        files_path.append("Database/"+bank_name+"/"+year+"/"+st.selectbox(label=" ", options=tables_input,label_visibility="collapsed",key=f"f{bank_num}"))
        if num == 1:
            tables_input2 =  [i for i in os.listdir("Database/"+bank_name+"/"+year2)] 
            files_path.append("Database/"+bank_name+"/"+year2+"/"+st.selectbox(label=" ", options=tables_input2,label_visibility="collapsed",key=f"f{bank_num}2"))
    return files_path
def get_files(num2):
    files = []
    for i in range(num2):
        files.append(st.file_uploader(label=" ",type=["csv","xlsx"],label_visibility="collapsed",key=f"file{i}"))
    return files

def readfile(path):
    if path.endswith("csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)        
    # df.columns = [i for i in range(len(df.columns))]
    df = df.where(pd.notnull(df), "-")
    return df
    
    
header(st,1,"مقارنة القوائم المالية ",font_size="30px",text_align="center")
st.markdown("---")

    
from_files = st.toggle(" استيراد القوائم بشكل يدوي ",key="toggle",value=False)
if from_files:
    col2, col1 = st.columns(2)
    with col1:
        header(st,1,"قم بتحديد عدد القوائم",font_size="20px")
    with col2:
        num2 = st.number_input(label=" ",min_value=1,max_value=1000,value=1,label_visibility="collapsed",key="num2")
    files = get_files(num2)
    # print(len(files))
else:
    col2, col1 = st.columns(2)
    with col1:
        header(st,1,"قم بتحديد عدد المصارف",font_size="20px")
    with col2:
        num = st.number_input(label=" ",min_value=1,max_value=1000,value=1,label_visibility="collapsed")
    file_paths = compare(num)

if st.toggle("مقارنة"):
    if not from_files:
        header(st,1,"قم بتحديد البيانات المراد مقارنتها",font_size="20px")
        dfs = []
        columns = []
        for i in range(len(file_paths)):
            df = readfile(file_paths[i])
            df = df.where(pd.notnull(df), "")
            df = df.astype(str)
            dfs.append(df)
            columns.append(st.multiselect(label=" ",options=df.columns,label_visibility="collapsed",key=f"col{i}"))

        new_df = pd.DataFrame()
        for i in range(len(dfs)):
            new_df = pd.concat([new_df,dfs[i][columns[i]]],axis=1)
        if not new_df.empty:
            new_df.columns = [i for i in range(len(new_df.columns))]
            new_df = new_df.where(pd.notnull(new_df), "")
            st.dataframe(new_df,use_container_width=True)
            st_profile_report(ProfileReport(new_df, title=f"المقارنة"))
        

    else:
        if len(files) > 0:
            header(st,1,"قم بتحديد البيانات المراد مقارنتها",font_size="20px")
            dfs = []
            columns = []
            for i in range(len(files)):
                try:
                    if str(files[i].type).endswith("csv"):
                        df = pd.read_csv(files[i])
                    else:
                        df = pd.read_excel(files[i])
                    df.columns = [i for i in range(len(df.columns))]
                    # convert all type to string
                    df = df.where(pd.notnull(df), "")
                    df = df.astype(str)
                    dfs.append(df)
                    columns.append(st.multiselect(label=" ",options=df.columns,label_visibility="collapsed",key=f"col{i}"))
                except:
                    pass
 
            
            new_df = pd.DataFrame()
            for i in range(len(dfs)):
                new_df = pd.concat([new_df,dfs[i][columns[i]]],axis=1)
            if not new_df.empty:
                new_df.columns = [i for i in range(len(new_df.columns))]
                new_df = new_df.where(pd.notnull(new_df), "")
                st.dataframe(new_df,use_container_width=True)
                st_profile_report(ProfileReport(new_df, title=f"المقارنة"))
            
st.markdown("---")
