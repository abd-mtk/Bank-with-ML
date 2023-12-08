import streamlit as st
import os
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pickle
from styles import style, header


st.set_page_config(
    page_title="برنامج المراقبة المالية المصرفي",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.image("images\\testing.png", width=150)
    header(st,1,"تجربة النماذج المعدة مسبقا ومعرفة النتائج المطلوبة",font_size="25px",font_weight="bold",text_align="center")
    st.markdown("---")
    
style(st)
@st.cache_data
def get_models():
    return [f"models/{model}" for model in os.listdir("models")]

header(st,1,"تجربة نماذج التوقع والتصنيف المحفوظة",font_size="30px",text_align="center")
st.markdown("---")
header(st,1,"النماذج المتوفرة",font_size="18px",text_align="right")
model_path = st.selectbox(label=" ", options=get_models(),label_visibility="collapsed")
model = pickle.load(open(model_path,"rb"))
header(st,1,f"{model_path.split('/')[1].split('_')[0].split('(')[1].replace(')','')}   :   تم تدريب هذه الخوارزمية على البيانات ",font_size="20px",text_align="right")
header(st,1,f"{model_path.split('/')[1].split('_')[1]}  :  اسم الخوارزمية المسخدمة",font_size="20px",text_align="right")
header(st,1,f"{model_path.split('/')[1].split('_')[3]} : المدخلات الواردة",font_size="20px",text_align="right")
st.markdown("---")
input_shape = int(model_path.split('/')[1].split('_')[3])
header(st,1,"مدخلات النموذج",font_size="20px",text_align="center",padding="10px")
inputs = []
for i in range(input_shape):
    inputs.append(st.text_input(label=" ",label_visibility="collapsed",key=f"input{i}"))
    
if st.button("تجربة النموذج"):
    st.markdown("---")
    x = np.array(inputs).astype(float)
    header(st,1,"نتيجة النموذج",font_size="20px",text_align="center")
    st.success(f"{model.predict([x])[0]}")
        