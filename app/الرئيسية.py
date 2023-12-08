import streamlit as st

st.set_page_config(
    page_title="برنامج المراقبة المالية المصرفي",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)


def header(st, h, text, text_color="#fff", text_align="right", font_size="15px", font_weight="bold", padding="0px"):
    return st.markdown(f"""<h{h} style='text-align: {text_align}; color: {text_color}; font-size: {font_size} ;font-weight: {font_weight}; padding: {padding}'>{text}</h{h}>""", unsafe_allow_html=True)

def style(st):
    return st.markdown(
        """
                <link href="//db.onlinewebfonts.com/c/6b75b24d502dab23003320c2e1b2fc68?family=Adobe+Arabic" rel="stylesheet" type="text/css"/>
                <style>
                    .stButton button {
                        background-color: #fff;
                        color: black;
                        padding: 10px 20px;
                        border-radius: 15px;
                        cursor: pointer;
                        width: 100%;
                        font-size: 20px;
                        font-weight: bold;
                    }
                    </style>
                    """,
        unsafe_allow_html=True,
    )

style(st)
with st.sidebar:
    st.image("images\\accounting.png", width=150)
    header(st, 1, "برنامج المراقبة المالية المصرفي",
           font_size="25px", font_weight="bold", text_align="center")
    st.markdown("---")

header(st, 1, "مشروع المراقبة والتحليل للبيانات المصرفية المتقدم",
       font_size="30px", font_weight="bold", text_align="center")
# st.markdown("---")
# header(st, 1, "المشروع مقدم من قبل الباحث ",
#        font_size="20px", text_align="center")
st.markdown("---")
header(st, 1, "يحتوي المشروع على   5  قوائم رئيسية وهي",
       font_size="20px", text_align="center")
st.markdown("---")

header(st, 1, "ادخال البيانات وانشاء الجداول",
    font_size="20px", text_align="right")
st.image("images\\edit.png", width=150)
st.markdown("---")

header(st, 1, "استكشاف وتصفح البيانات",
    font_size="20px", text_align="right")
st.image("images\\exploration.png", width=150)
st.markdown("---")

header(st, 1, "مقارنة البيانات", font_size="20px", text_align="right")
st.image("images\\compare.png", width=150)
st.markdown("---")

header(st, 1, "نمذجة البيانات وتوقع النتائج بواسطة الذكاء الاصطناعي",
    font_size="20px", text_align="right")
st.image("images\\ai.png", width=150)

st.markdown("---")
header(st, 1, "استخدام النماذج المتقدمة",
    font_size="20px", text_align="right")
st.image("images\\testing.png", width=150)

