# Demo 5: Mini-Gapminder Style Dashboard with Streamlit

This demo focuses on creating a more dynamic, Gapminder-inspired interactive dashboard using Streamlit and Altair.

**Task:**
1.  Use an appropriate dataset (e.g., public Gapminder data or a suitable health equivalent with time-series data for multiple entities, including variables for X-axis, Y-axis, Size, and Color).
2.  Implement a `st.slider` for "Year" selection.
3.  Create an Altair chart with encodings for X, Y, Size, and Color, designed to be filtered by the selected year.
4.  Dynamically filter the DataFrame passed to the Altair chart based on the Streamlit slider's value.
5.  Display the interactive Altair chart using `st.altair_chart()`.