import streamlit as st
import pandas as pd
import numpy as np

st.title("Welcome to the CHM Frontend")

st.write("This is a simple Streamlit application.")

data = np.random.randn(10, 2)
df = pd.DataFrame(data, columns=["Column 1", "Column 2"])

st.line_chart(df)
