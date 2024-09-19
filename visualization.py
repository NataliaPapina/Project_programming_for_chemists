import plotly.express as px
import plotly.graph_objects as go
from preprocessing import data, data_normalized


def bar(df, column):
    fig = px.bar(df, x="class", y=column)
    fig.show()


def heatmap(df):
    fig = px.imshow(df.corr(), title='Correlation heatmap')
    fig.show()


def box_plot(df, name):
    fig = go.Figure()
    for i in df.columns:
        fig.add_trace(go.Box(y=df[i], name=i))
    fig.update_layout(title_text=name, title_x=0.5)
    fig.show()


def one_box_plot(df, name):
    fig = px.box(df, x="class", y=name)
    fig.show()


for i in data.drop(['viability'], axis=1).columns:
    if data[i].dtype == 'float64' or data[i].dtype == 'int64':
        one_box_plot(data, i)

box_plot(data_normalized.drop(['NPs', 'Cellline', 'Celltype', 'class'], axis=1), 'before normalization')