import streamlit as st
import os
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from AutoClean import AutoClean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from styles import style, header


st.set_page_config(
    page_title="برنامج المراقبة المالية المصرفي",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.image("images\\ai.png", width=150)
    header(st,1,"استخدام تقنيات الذكاء الاصطناعي في بناء نماذج توقع وتصنيف السلوك المالي والمخاطر المستقبلية لتعزيز الاحصائيات بشكل متقدم",font_size="25px",font_weight="bold",text_align="center")
    st.markdown("---")
    
style(st)
@st.cache_data
def get_bankes(): return os.listdir("Database")
@st.cache_data
def get_years(bank_name): return os.listdir("Database/"+bank_name)


@st.cache_data
def const_names(task):
    algoclf = ["DecisionTreeClassifier",
               "RandomForestClassifier", "AdaBoostClassifier"]
    algoreg = ["DecisionTreeRegressor",
               "RandomForestRegressor", "AdaBoostRegressor"]
    if task == "تصنيف":
        return algoclf
    else:
        return algoreg


header(st, 1, "مميزات متقدمة في تحليل وتوقع السلوك المالي وتصنيف البيانات بواسطة تقنيات الذكاء الاصطناعي",
       font_size="25px", text_align="center")
st.markdown("---")


def compare(num):
    files_path = []
    for bank_num in range(num):
        col1, col2, col3 = st.columns(3)
        with col1:
            bank_name = st.selectbox(options=get_bankes(
            ), label_visibility="collapsed", label=" ", key=f"b{bank_num}")
        with col2:
            header(st, 1, bank_name, font_size="20px")
        with col3:
            header(st, 1, "ادارة مصرف", font_size="20px")
        col1, col2, col3 = st.columns(3)
        with col1:
            year = str(st.selectbox(options=get_years(
                bank_name), label_visibility="collapsed", label=" ", key=f"y{bank_num}"))
        with col2:
            header(st, 1, year, font_size="20px")
        with col3:
            header(st, 1, "الحسابات للسنة المالية", font_size="20px")

        tables_input = [i for i in os.listdir("Database/"+bank_name+"/"+year)]
        header(st, 1, "القوائم المتوفرة", font_size="20px")
        files_path.append("Database/"+bank_name+"/"+year+"/"+st.selectbox(label=" ",
                          options=tables_input, label_visibility="collapsed", key=f"f{bank_num}"))
    return files_path


def get_files(num2):
    files = []
    for i in range(num2):
        files.append(st.file_uploader(label=" ", type=[
                     "csv", "xlsx"], label_visibility="collapsed", key=f"file{i}"))
    return files


def readfile(path):
    if path.endswith("csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df = df.where(pd.notnull(df), "")
    return df

def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="predicted label", ylabel="true label")
  
def train_model(model, x_train, y_train,prefix=""):
    name = "Data("+prefix+")_"+model
    if model == "DecisionTreeClassifier":
        model = DecisionTreeClassifier()
    elif model == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model == "AdaBoostClassifier":
        model = AdaBoostClassifier()
    elif model == "DecisionTreeRegressor":
        model = DecisionTreeRegressor()
    elif model == "RandomForestRegressor":
        model = RandomForestRegressor()
    elif model == "AdaBoostRegressor":
        model = AdaBoostRegressor()

    st.spinner("جاري التدريب")
    model = model.fit(x_train, y_train)
    st.success("تم التدريب بنجاح")
    acc = accuracy_score(y_train, model.predict(x_train))
    st.info(f"النسبة المئوية لدقة للتدريب {acc}")
    cm = confusion_matrix(y_train, model.predict(x_train))
    conf = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=model.classes_, yticklabels=model.classes_,fmt='.0f')
    st.pyplot(conf.figure)
    pickle.dump(model, open(f"models\{name}_inputShape_{x_train.shape[1]}_model.pkl", "wb"))
    


if st.toggle(" استيراد القوائم بشكل يدوي ", key="toggle"):
    file = st.file_uploader(label=" ", type=[
                     "csv", "xlsx"], label_visibility="collapsed", key="file")
    if file is not None:
        try:

            if str(file.type).endswith("csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
                df = df.where(pd.notnull(df), np.nan)
            if st.toggle("اظهار البيانات ", key="show"):
                st.dataframe(df, use_container_width=True)
            header(st, 1, "قم بتحديد البيانات  المطلوب تحليلها",
                    font_size="20px")
            columns = st.multiselect(
                label=" ", options=df.columns, label_visibility="collapsed", key="col")
            header(
                st, 1, "قم بتحديد البيانات  المطلوب توقعها او تصنيفها", font_size="20px")
            y = st.selectbox(label=" ", options=df.columns,
                        label_visibility="collapsed", key="lable")
            header(st, 1, "قم بتحديد الصفوف غير المطلوبة ", font_size="20px")
            drop_rows = st.multiselect(
                label=" ",
                options=df.index,
                label_visibility="collapsed",
            )
            new_df = pd.DataFrame()
            new_df = pd.concat([new_df, df[columns]], axis=1)
            new_df = new_df.where(pd.notnull(new_df), np.nan)
            new_df.dropna(axis=1, how="all", inplace=True)
            new_df.drop_duplicates(inplace=True)
            new_df = new_df[~new_df.index.isin(drop_rows)]
            df.where(pd.notnull(df), np.nan)
            df.dropna(axis=1, how="all", inplace=True)
            df.drop_duplicates(inplace=True)
            
            if st.toggle("اظهار البيانات المطلوبة ", key="show2"):
                if y in new_df.columns:
                    x_show = new_df.drop([y], axis=1)
                else:
                    x_show = new_df
                y_show = pd.DataFrame(df[y])
                col1, col2 = st.columns(2)
                with col1:
                    x_show.columns = [i for i in range(len(x_show.columns))]
                    st.dataframe(x_show, use_container_width=True)
                with col2:
                    y_show.columns = [i for i in range(len(y_show.columns))]
                    st.dataframe(y_show, use_container_width=True)
                
            col1, col2 = st.columns(2)
            with col2:
                header(st, 1, "قم بتحديد نوع التدريب ", font_size="20px")
                task = st.selectbox(
                    options=["تصنيف", "توقع"], label=" ", key="task",label_visibility="collapsed")
            with col1:
                header(st, 1, "قم بتحديد الخوارزمية ", font_size="20px")
                algo = st.selectbox(
                    options=const_names(task), label=" ", key="algo",label_visibility="collapsed")
           
            if st.toggle("تحليل البيانات", key="analyze"):
                if not new_df.empty:
                    st.dataframe(new_df, use_container_width=True)
                    st_profile_report(ProfileReport(new_df, title=f"المقارنة"))
                else:
                    st.error("لا يوجد بيانات")
            if st.button(label="بداء", key="train"):
                if y in new_df.columns:
                    x = new_df.drop([y], axis=1)
                else:
                    x = new_df
                y = pd.DataFrame(df[y])
                train_model(algo, x, y,file.name.split(".")[0])
                
        except Exception as e:
            st.error("حدث خطاء يرجى التحقق من صحة البيانات")
            st.error(e)
            pass

else:
    try:
        file_paths = compare(1)
        df = readfile(file_paths[0])
        df = df.where(pd.notnull(df), np.nan)
        header(st, 1, "قم بتحديد البيانات  المطلوب تحليلها", font_size="20px")
        columns = st.multiselect(label=" ", options=df.columns,
                    label_visibility="collapsed", key="col")
        header(st, 1, "قم بتحديد البيانات  المطلوب توقعها او تصنيفها",
            font_size="20px")
        y = st.selectbox(label=" ", options=df.columns,
                label_visibility="collapsed", key="lable")

        header(st, 1, "قم بتحديد الصفوف غير المطلوبة ", font_size="20px")
        drop_rows = st.multiselect(
            label=" ",
            options=df.index,
            label_visibility="collapsed",
        )
        new_df = pd.DataFrame()
        new_df = pd.concat([new_df, df[columns]], axis=1)
        new_df = new_df.where(pd.notnull(new_df), np.nan)
        new_df.dropna(axis=1, how="all", inplace=True)
        new_df.drop_duplicates(inplace=True)
        new_df = new_df[~new_df.index.isin(drop_rows)]
        df.where(pd.notnull(df), np.nan)
        df.dropna(axis=1, how="all", inplace=True)
        df.drop_duplicates(inplace=True)
        
        if st.toggle("اظهار البيانات المطلوبة ", key="show2"):
            if y in new_df.columns:
                x_show = new_df.drop([y], axis=1)
            else:
                x_show = new_df
            y_show = pd.DataFrame(df[y])
            col1, col2 = st.columns(2)
            with col1:
                x_show.columns = [i for i in range(len(x_show.columns))]
                st.dataframe(x_show, use_container_width=True)
            with col2:
                y_show.columns = [i for i in range(len(y_show.columns))]
                st.dataframe(y_show, use_container_width=True)
            
        col1, col2 = st.columns(2)
        with col2:
            header(st, 1, "قم بتحديد نوع التدريب ", font_size="20px")
            task = st.selectbox(
                options=["تصنيف", "توقع"], label=" ", key="task",label_visibility="collapsed")
        with col1:
            header(st, 1, "قم بتحديد الخوارزمية ", font_size="20px")
            algo = st.selectbox(
                options=const_names(task), label=" ", key="algo",label_visibility="collapsed")
        
        if st.toggle("تحليل البيانات", key="analyze"):
            if not new_df.empty:
                st.dataframe(new_df, use_container_width=True)
                st_profile_report(ProfileReport(new_df, title=f"المقارنة"))
            else:
                st.error("لا يوجد بيانات")
        if st.button(label="بداء", key="train"):
            if y in new_df.columns:
                x = new_df.drop([y], axis=1)
            else:
                x = new_df
            y = pd.DataFrame(df[y])
            train_model(algo, x, y,file_paths[0].split("/")[-1].split(".")[0])
            
    except Exception as e:
        st.error("حدث خطاء يرجى التحقق من صحة البيانات")
        st.error(e)
        pass
st.markdown("---")
