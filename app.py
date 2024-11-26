import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load results
original_results = pd.read_csv('model_comparison.csv')
optimized_results = pd.read_csv('optimized_model_comparison.csv')

# Function to encode images
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'

# Layout
app.layout = dbc.Container([
    html.H1("Diabetes Prediction Model Analysis", className="text-center my-4"),

    # Dataset Overview
    dbc.Card([
        dbc.CardHeader(html.H3("Dataset Overview")),
        dbc.CardBody([
            html.P([
                "The analysis was performed on the Pima Indians Diabetes dataset, which contains various health metrics ",
                "and a binary outcome indicating the presence of diabetes. The dataset includes 768 instances with 8 features:"
            ]),
            html.Ul([
                html.Li("Pregnancies"),
                html.Li("Glucose"),
                html.Li("Blood Pressure"),
                html.Li("Skin Thickness"),
                html.Li("Insulin"),
                html.Li("BMI"),
                html.Li("Diabetes Pedigree Function"),
                html.Li("Age")
            ])
        ])
    ], className="mb-4"),

    # Feature Importance
    dbc.Card([
        dbc.CardHeader(html.H3("Feature Importance Analysis")),
        dbc.CardBody([
            html.P("The Random Forest algorithm was used to determine the most important features:"),
            html.Img(src=encode_image('feature_importance.png'), className="img-fluid")
        ])
    ], className="mb-4"),

    # Model Comparison
    dbc.Card([
        dbc.CardHeader(html.H3("Model Comparison")),
        dbc.CardBody([
            html.H4("Original Models (All Features)", className="mb-3"),
            dbc.Table.from_dataframe(original_results, striped=True, bordered=True, hover=True),

            html.H4("Optimized Models (Selected Features)", className="mt-4 mb-3"),
            dbc.Table.from_dataframe(optimized_results, striped=True, bordered=True, hover=True),

            html.P([
                "The optimized models use only the most important features: ",
                "Glucose, BMI, Diabetes Pedigree Function, and Age."
            ], className="mt-3")
        ])
    ], className="mb-4"),

    # ROC Curves
    dbc.Card([
        dbc.CardHeader(html.H3("ROC Curves")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4("Original Models"),
                    html.Img(src=encode_image('roc_curve_random_forest.png'), className="img-fluid"),
                    html.Img(src=encode_image('roc_curve_svm.png'), className="img-fluid"),
                    html.Img(src=encode_image('roc_curve_logistic_regression.png'), className="img-fluid")
                ], width=6),
                dbc.Col([
                    html.H4("Optimized Models"),
                    html.Img(src=encode_image('optimized_roc_curve_random_forest.png'), className="img-fluid"),
                    html.Img(src=encode_image('optimized_roc_curve_svm.png'), className="img-fluid"),
                    html.Img(src=encode_image('optimized_roc_curve_logistic_regression.png'), className="img-fluid")
                ], width=6)
            ])
        ])
    ], className="mb-4"),

    # Key Findings
    dbc.Card([
        dbc.CardHeader(html.H3("Key Findings")),
        dbc.CardBody([
            html.Ul([
                html.Li("Glucose level is the most important feature for diabetes prediction"),
                html.Li("The optimized models using only 4 features perform similarly to models using all features"),
                html.Li("SVM and Logistic Regression achieved the highest accuracy (76.62%) with the reduced feature set"),
                html.Li("The models show good discrimination ability with AUC values around 0.81-0.83")
            ])
        ])
    ], className="mb-4")
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
