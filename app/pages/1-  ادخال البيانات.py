import os
import numpy as np
import pandas as pd
import streamlit as st

from styles import style,header


st.set_page_config(
    page_title="برنامج المراقبة المالية المصرفي",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
    )


with st.sidebar:
    st.image("images\\edit.png", width=150)
    header(st,1,"ادخال البيانات بسهولة وسرعة وفق الانماط الثابته او انشاء جداول جديدة واضفاة البيانات عليها",font_size="25px",font_weight="bold",text_align="center")
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
    year = str(st.selectbox(options=get_years(bank_name),label_visibility="collapsed",label= " "))
with col2:
    header(st,1,year,font_size="20px")
with col3:
    header(st,1,"الحسابات للسنة المالية",font_size="20px")
st.markdown("---")

    
    
tables_input =  [i for i in os.listdir("Database/"+bank_name+"/"+year)] 

col1, col2 = st.columns(2)
with col1:
    header(st,1," قائمة الادخال المخصصة",font_size="20px")
    file_name = st.text_input(label=" ",placeholder="عنوان القائمة الجديدة")
    header(st,1,"حدد عدد الاعمدة",font_size="20px",text_align="center")
    col_num = st.number_input(label=" ",min_value=1,max_value=1000,value=1,label_visibility="collapsed")
with col2:
    header(st,1,"قوائم الادخال الثابتة",font_size="20px")
    fixed_type = st.selectbox(label=" ", options=tables_input)
    header(st,1,"حدد القائمة لادخال البيانات او قم باختيار الملف",font_size="20px",text_align="center")
    # file_path = st.file_uploader(label="",type=["csv","xlsx"],label_visibility="collapsed")

input_type = st.selectbox(label=" ", options=["مخصصة","ثابتة"])
add_mode = st.toggle("بدا",key="mode")
st.markdown("---")

def create_column_form(col_num,lables=[]):
    try:
        rows = []
        count = 0
        if len(lables) == 0:
            for j in range(col_num):
                if col_num > 2 and j < col_num-1 and count < col_num-1:
                    t1, t2 = st.columns(2)
                    with t1:
                        header(st,1,COLUMN_NAME+str(count+1)+" ",font_size="16px")
                        rows.append(st.text_input(label=" ", key=f"k1{j}"))
                        count += 1
                    with t2:
                        header(st,1,COLUMN_NAME+str(count+1)+" ",font_size="16px")
                        rows.append(st.text_input(label=" ", key=f"k2{j}", label_visibility="collapsed"))
                        count += 1
                elif count <= col_num-1:
                    header(st,1,COLUMN_NAME+str(count+1)+" ",font_size="16px")
                    rows.append(st.text_input(label=" ", key=f"k1{j}", label_visibility="collapsed"))
                    count += 1
        else:
            for j in range(col_num):
                if col_num > 2 and j < col_num-1 and count < col_num-1:
                    t1, t2 = st.columns(2)
                    with t1:
                        header(st,1,str(lables[count])+" ",font_size="16px")
                        rows.append(st.text_input(label=" ", key=f"k1{j}", label_visibility="collapsed"))
                        count += 1
                    with t2:
                        header(st,1,str(lables[count])+" ",font_size="16px")
                        rows.append(st.text_input(label=" ", key=f"k2{j}", label_visibility="collapsed"))
                        count += 1
                elif count <= col_num-1:
                    header(st,1,str(lables[count])+" ",font_size="16px")
                    rows.append(st.text_input(label=" ", key=f"k1{j}", label_visibility="collapsed"))
                    count += 1        
        return rows
    except Exception as e:
        print(e)
        return []


if add_mode and input_type == "مخصصة":
    data = create_column_form(col_num)
    lables = []
    for i in data:
        if i != "" and i not in lables:
            lables.append(i)
    if st.button("انشاء قائمة جديدة"):
        if len(lables) == 0 or len(lables) != col_num:
            st.error("يجب اضافة عنوان لكل عامود")
            st.stop()
        df = pd.DataFrame(columns=lables)
        st.dataframe(df,use_container_width=True)
        if file_name == "":
            st.error("يجب ادخال عنوان القائمة الجديدة")
            st.stop()
        path = "Database/"+bank_name+"/"+year
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+"/"+file_name+".csv"
        df.to_csv(path,index=False)
        st.success("تم اضافة القائمة بنجاح")
        st.stop()  
        
elif add_mode and input_type == "ثابتة":
    path = "Database/"+bank_name+"/"+year+"/"+fixed_type
    if path.endswith(".csv"):
        orginal = pd.read_csv(path)
        row = create_column_form(len(orginal.columns),orginal.columns)
        new_data = pd.DataFrame([row],columns=orginal.columns)
        st.dataframe(new_data,use_container_width=True)
        st.markdown("---")
        if st.button("تاكيد الادخال",key="confirm csv"):
            df = pd.concat([orginal,new_data],ignore_index=True)
            df.to_csv(path,index=False)
            st.dataframe(df,use_container_width=True)
            st.success("تم اضافة البيانات بنجاح")
    else:
      orginal = pd.read_excel(path)
      orginal.columns = [i for i in range(len(orginal.columns))]
      orginal = orginal.where(pd.notnull(orginal), "")
      row = create_column_form(len(orginal.columns),orginal.columns)
      new_data = pd.DataFrame([row],columns=orginal.columns)
      st.dataframe(new_data,use_container_width=True)
      st.markdown("---")
      if st.button("تاكيد الادخال",key="confirm xlsx"):
        df = pd.concat([orginal,new_data],ignore_index=True)
        df.to_excel(path,index=False)
        st.dataframe(df,use_container_width=True)
        st.success("تم اضافة البيانات بنجاح")
      
        
    
