# Demo 4: Simple, Practical Health Dashboard with Streamlit

This demo focuses on building a basic, functional Streamlit dashboard for data exploration.

**Task:**
1.  Load a small PhysioNet dataset snippet (e.g., patient demographics, lab values).
2.  Use `st.sidebar` to create filters (e.g., `st.selectbox` for a categorical variable, `st.slider` for a numerical range).
3.  Filter a Pandas DataFrame based on these selections.
4.  Display the filtered DataFrame using `st.dataframe()`.
5.  Display a simple Altair chart based on the filtered data using `st.altair_chart()`.
6.  Show a key summary statistic using `st.metric()`.