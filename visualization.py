import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc


def hist(df, column):
    fig = px.histogram(df, x=column)
    fig.show()


def bar(df, column):
    fig = px.bar(df, x="class", y=column)
    fig.show()


def corr_heatmap(df, text=False):
    fig = px.imshow(df.corr(method='spearman', numeric_only=True), title='Correlation heatmap', text_auto=text)
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


def roc_(fpr, tpr):
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=800, height=800
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


def scatter(x, y, yaxes='test', xaxes='test_pred'):
    fig = px.scatter(x=x, y=y)
    fig.update_yaxes(title_text=yaxes)
    fig.update_xaxes(title_text=xaxes)
    fig.show()