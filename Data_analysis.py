import plotly.express as px
from preprocessing import data
from Multicollinearity_test import data_normalized, df_independent_variables
import plotly.graph_objects as go

fig0 = px.bar(data, x="class", )
fig0.show()

fig = px.imshow(df_independent_variables.corr(),title='Сorrelation heatmap')
fig.show()

fig2 = go.Figure()
for i in data_normalized.columns:
    fig2.add_trace(go.Box(y=data[i], name=i))
fig2.update_layout(title_text='Box plot before normalization', title_x=0.5)
fig2.show()

fig3 = go.Figure()
for i in data_normalized.columns:
    fig3.add_trace(go.Box(y=data_normalized[i], name=i))
fig3.update_layout(title_text='Box plot after normalization', title_x=0.5)
fig3.show()