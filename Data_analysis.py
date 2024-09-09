import pandas as pd
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
from preprocessing import data, data_normalized, independent_variables, independent_variables_normalized
import plotly as py
from plotly import express
import numpy as np
import plotly.graph_objects as go
from plotly.colors import n_colors

fig = px.imshow(independent_variables.corr(),title='Сorrelation heatmap')
fig.show()

fig2 = go.Figure()
for i in independent_variables.columns:
    fig2.add_trace(go.Box(y=data[i], name=i))
fig2.update_layout(title_text='Box plot before normalization', title_x=0.5)
fig2.show()

fig3 = go.Figure()
for i in independent_variables.columns:
    fig3.add_trace(go.Box(y=data_normalized[i], name=i))
fig3.update_layout(title_text='Box plot after normalization', title_x=0.5)
fig3.show()