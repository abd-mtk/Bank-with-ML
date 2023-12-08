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
            .stSlider {
                    background-color: #123;
                    padding: 10px;
                    border-radius: 15px;
                    display: inline-flex;
                    cursor: pointer;
        }

                    </style>
                    """,
        unsafe_allow_html=True,
    )
def header(st,h,text,text_color="#fff",text_align="right",font_size="15px",font_weight="bold",padding="0px"):
    return st.markdown(f"""<h{h} style='text-align: {text_align}; color: {text_color}; font-size: {font_size} ;font-weight: {font_weight}; padding: {padding}'>{text}</h{h}>""", unsafe_allow_html=True)
