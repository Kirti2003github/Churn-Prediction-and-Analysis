import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import sqlite3
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer  , OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')
def register_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()

def authenticate_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    return c.fetchone() is not None

username=""
# Login and registration forms
def show_login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state['authenticated'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Login failed")
    return username

def show_register():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        register_user(username, password)
        st.success("User registered successfully!")

# Main app logic
if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    choice = st.selectbox("Choose an option", ["Login", "Register"])
    if choice == "Login":
        username=show_login()
    else:
        show_register()
else:
    st.write("You are logged in")
    # Show your main app content
# Sidebar file uploader and navigation
    with st.sidebar:
        st.markdown("<h2 style='font-size: 26px; font-weight: bold;'>Upload your file</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("")

        # Buttons for navigation
        if st.button('Dashboard'):
            st.session_state.page = 'dashboard'
        if st.button('ML Model'):
            st.session_state.page = 'ml_model'

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'

    # Default content for the sidebar
    if uploaded_file is None:
        info_content = """
        File Features:
        - CustomerID
        - Churn
        - Tenure
        - PreferredLoginDevice
        - CityTier
        - WarehouseToHome
        - PreferredPaymentMode
        - Gender
        - HourSpendOnApp
        - NumberOfDeviceRegistered
        - PreferedOrderCat
        - SatisfactionScore
        - MaritalStatus
        - NumberOfAddress
        - Complain
        - OrderAmountHikeFromlastYear
        - CouponUsed
        - OrderCount
        - DaySinceLastOrder
        - CashbackAmount
    
        The uploaded file contains various features that will be used for churn analysis and generating valuable insights. Get ready to explore the data and uncover actionable information to drive business decisions.
        """

        def create_download_link(content, filename):
            b64 = base64.b64encode(content.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Features Information</a>'
            return href

        st.markdown(
            """
            <div style='background-color: #f9f9f9; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
                <h3 style='color: #333333; font-size: 20px; font-weight: bold; margin-bottom: 15px;'>Why Churn Analysis is Important?</h3>
                <p style='color: #555555; font-size: 16px; margin-bottom: 10px;'>Churn analysis is crucial for understanding customer behavior and improving retention strategies.</p>
                <h3 style='color: #333333; font-size: 20px; font-weight: bold; margin-bottom: 15px;'>Churn Rate Formula:</h3>
                <p style='color: #555555; font-size: 16px; margin-bottom: 10px;'>Churn Rate = (Customers Lost / Customers at Start) x 100</p>
                <h3 style='color: #333333; font-size: 20px; font-weight: bold; margin-bottom: 15px;'>Get Started:</h3>
                <p style='color: #555555; font-size: 16px; margin-bottom: 10px;'>Upload your data file to analyze churn patterns and make informed decisions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.write('Click on the link below before uploading your file')
        st.markdown(create_download_link(info_content, "features_info.txt"), unsafe_allow_html=True)
        st.stop()

    @st.cache_data
    def load_data(path: str):
        df = pd.read_excel(path)
        return df

    if uploaded_file:
        df = load_data(uploaded_file)
        df.drop('CustomerID', inplace=True, axis=1)
        df1=df.copy()
        cat = df.select_dtypes(include='object').columns
        num = list(df.select_dtypes(exclude='object').columns)
        num.remove('Churn')
        for cols in num:
            Q1 = df[cols].quantile(0.25)
            Q3 = df[cols].quantile(0.75)
            IQR = Q3 - Q1
            lr = Q1 - (1.5 * IQR)
            ur = Q3 + (1.5 * IQR)
            df[cols] = df[cols].mask(df[cols] < lr, lr, )
            df[cols] = df[cols].mask(df[cols] > ur, ur, )

        df['Tenure'].fillna(df.Tenure.median(), inplace=True)
        df['WarehouseToHome'].fillna(df.WarehouseToHome.median(), inplace=True)
        df['HourSpendOnApp'].fillna(df.HourSpendOnApp.median(), inplace=True)
        df['OrderAmountHikeFromlastYear'].fillna(round(df.OrderAmountHikeFromlastYear.mean()), inplace=True)
        df['CouponUsed'].fillna(df.CouponUsed.median(), inplace=True)
        df['OrderCount'].fillna(df.OrderCount.median(), inplace=True)
        df['DaySinceLastOrder'].fillna(df.DaySinceLastOrder.median(), inplace=True)
        enc = LabelEncoder()
        for col in df.select_dtypes(include='object'):
            df[col] = enc.fit_transform(df[col])
        X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df.Churn)

        class my_classifier(BaseEstimator):
            def __init__(self, estimator=None):
                self.estimator = estimator

            def fit(self, X, y=None):
                self.estimator.fit(X, y)
                return self

            def predict(self, X, y=None):
                return self.estimator.predict(X, y)

            def predict_proba(self, X):
                return self.estimator.predict_proba(X)

            def score(self, X, y):
                return self.estimator.score(X, y)

        categorical_features = [1, 4, 5, 8, 10]
        numeric_features = [0, 2, 3, 6, 7, 9, 11, 12, 13, 14, 15, 16]

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', my_classifier())
        ])

        parameters = [
            {'clf': [LogisticRegression(max_iter=1000)],
             'clf__C': [0.001, 0.01, .1, 1],
             'clf__solver': ['lbfgs', 'liblinear']
             },
            {'clf': [RandomForestClassifier()],
             'clf__criterion': ['gini', 'entropy'],
             },
            {
                'clf': [DecisionTreeClassifier()],
                'clf__criterion': ['gini', 'entropy'],
            },
            {
                'clf': [XGBClassifier()],
                'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'clf__reg_lambda': [0.01, 0.1, 1],
                'clf__reg_alpha': [0.01, 0.1, 0, 1],
            }]

        grid = GridSearchCV(pipe, parameters, cv=5)
        grid.fit(X_train, y_train)
        joblib.dump(grid, 'xgbpipe.joblib')
        model = joblib.load('xgbpipe.joblib')

        if st.session_state.page == 'dashboard':
            # Your dashboard code here
            #st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
            #st.write("Add your dashboard code here.")
            df = load_data(uploaded_file)
            filename = uploaded_file.name
            st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
            image = Image.open('logo2.png')

            col1, col2, col11, col12, col13 = st.columns([0.17, 0.35, 0.16, 0.16, 0.16])
            with col1:
                st.image(image, width=280)

            html_title = """
                <style>
                .title-test {
                font-weight:bold;
                border-radius:6px;
    
                }
                </style>
                <h1 class="title-test">Ecommerce Churn Analysis Dashboard</h1>"""
            with col2:
                st.markdown(html_title, unsafe_allow_html=True)

            # def display_kpi_metrics(kpis: List[float], kpi_names: List[str]):

            #   for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(4), zip(kpi_names, kpis))):
            #      col.metric(label=kpi_name, value=kpi_value)
            with col11:
                col11.markdown("""<style>.metric h2 {font-size: 32px; color: black;}</style>""", unsafe_allow_html=True)
                col11.markdown("""<style>.metric {background-color: white;}</style>""", unsafe_allow_html=True)
                col11.metric(label='Churn Rate',
                             value=str(round(df[df['Churn'] == 1]['Churn'].value_counts()[1] * 100 / len(df), 2)) + '%')
                style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",
                                   box_shadow="#F71938")
            with col12:
                col12.markdown("""<style>.metric h2 {font-size: 32px; color: black;}</style>""", unsafe_allow_html=True)
                col12.markdown("""<style>.metric {background-color: white;}</style>""", unsafe_allow_html=True)
                col12.metric(label='Average Cashback Amount For Churned',
                             value=round(df['CashbackAmount'].mean(), 2))
                style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",
                                   box_shadow="#F71938")

            with col13:
                col13.markdown("""<style>.metric h2 {font-size: 32px; color: black;}</style>""", unsafe_allow_html=True)
                col13.markdown("""<style>.metric {background-color: white;}</style>""", unsafe_allow_html=True)
                col13.metric(label='Average Order Amount Hike per Customer',
                             value=round(df['OrderAmountHikeFromlastYear'].sum() / len(df), 2))
                style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",
                                   box_shadow="#F71938")

            col3, col4, col5, col6 = st.columns([0.25, 0.25, 0.25, 0.25])
            col7, col8, col9, col10 = st.columns([0.25, 0.25, 0.25, 0.25])
            with col3:
                quartiles = list(df['WarehouseToHome'].quantile([0.25, 0.5, 0.75]))
                print(quartiles)


                def fun(x):
                    if x <= quartiles[0]:
                        return 'Very Close Distance'
                    elif x > quartiles[0] and x <= quartiles[1]:
                        return 'Close Distance'
                    elif x > quartiles[1] and x <= quartiles[2]:
                        return 'Moderate Distance'
                    else:
                        return 'Far Distance'


                df['distance_from_warehouse'] = df['WarehouseToHome'].apply(lambda x: fun(x))
                churn_percentage = df[df['Churn'] == 1]['Churn'].groupby(df['distance_from_warehouse']).size() / df.groupby(
                    'distance_from_warehouse').size()

                # Convert the result into a DataFrame
                churn_percentage_df = churn_percentage.reset_index()

                # Rename columns for clarity
                churn_percentage_df.columns = ['distance_from_warehouse', 'churn_percentage']
                churn_percentage_df = churn_percentage_df.sort_values('churn_percentage', ascending=False)
                # Plot using Plotly Express
                fig = px.bar(churn_percentage_df, x='distance_from_warehouse', y='churn_percentage',
                             title='Churn Rate by Warehouse-To-Home',
                             color=churn_percentage_df['churn_percentage'], color_continuous_scale='blues',
                             color_continuous_midpoint=churn_percentage_df['churn_percentage'].median(), height=400,
                             width=1000, template="gridon")  # Use the median churn percentage as the color midpoint
                fig.update_traces(text=churn_percentage_df['churn_percentage'], texttemplate='%{text:.2%}',
                                  textposition='inside')
                fig.update_yaxes(tickformat=',.0%')
                fig.update_layout(yaxis_title='', xaxis_title='', coloraxis_showscale=False,
                                  plot_bgcolor='rgb(255, 255, 255)', paper_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                churn_percentage_ppm = df[df['Churn'] == 1].groupby('PreferredPaymentMode').size() / df.groupby(
                    'PreferredPaymentMode').size()

                # Convert the result into a DataFrame
                churn_percentage_ppm_df = churn_percentage_ppm.reset_index()

                # Rename columns for clarity
                churn_percentage_ppm_df.columns = ['PreferredPaymentMode', 'churn_percentage']
                churn_percentage_ppm_df = churn_percentage_ppm_df.sort_values('churn_percentage', ascending=True)
                # Plot using Plotly Express

                fig = px.bar(churn_percentage_ppm_df, x='churn_percentage', y='PreferredPaymentMode',
                             title='Churn Rate by Preferred Payment Mode',
                             color=churn_percentage_ppm_df['churn_percentage'], color_continuous_scale='blues',
                             color_continuous_midpoint=churn_percentage_ppm_df['churn_percentage'].median(), height=400,
                             width=1000)  # Use the median churn percentage as the color midpoint
                fig.update_traces(text=churn_percentage_ppm_df['churn_percentage'], texttemplate='%{text:.2%}',
                                  textposition='inside')
                fig.update_yaxes(tickformat=',.0%')
                fig.update_layout(yaxis_title='', xaxis_title='', coloraxis_showscale=False,
                                  plot_bgcolor='rgb(255, 255, 255)', paper_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig, use_container_width=True)

            with col5:
                quartiles_tenure = list(df['Tenure'].quantile([0.25, 0.5, 0.75]))
                print(quartiles)


                def fun(x):
                    if x <= quartiles_tenure[0]:
                        return 'new customers'
                    elif x > quartiles_tenure[0] and x <= quartiles_tenure[2]:
                        return 'established customers'
                    elif x > quartiles_tenure[2]:
                        return 'long term customers'


                df['TenureGroup'] = df['Tenure'].apply(lambda x: fun(x))
                churn_percentage = df[df['Churn'] == 1]['Churn'].groupby(df['TenureGroup']).sum() / len(
                    df[df['Churn'] == 1])
                churn_percentage_df = churn_percentage.reset_index()

                # Rename columns for clarity
                churn_percentage_df.columns = ['TenureGroup', 'churn_percentage']
                churn_percentage_df = churn_percentage_df.sort_values('churn_percentage', ascending=False)
                # Plot using Plotly Express
                fig = px.bar(churn_percentage_df, x='TenureGroup', y='churn_percentage',
                             title='Churn Rate by Tenure Group',
                             color=churn_percentage_df['churn_percentage'], color_continuous_scale='blues',
                             color_continuous_midpoint=churn_percentage_df['churn_percentage'].median(), height=400,
                             width=1000)  # Use the median churn percentage as the color midpoint
                fig.update_traces(text=churn_percentage_df['churn_percentage'], texttemplate='%{text:.2%}',
                                  textposition='inside')
                fig.update_yaxes(tickformat=',.0%')
                fig.update_layout(yaxis_title='', xaxis_title='', coloraxis_showscale=False,
                                  plot_bgcolor='rgb(255, 255, 255)', paper_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig, use_container_width=True)

            with col6:
                churn_percentage_CT = df[df['Churn'] == 1]['Churn'].groupby(df['CityTier']).size() / df.groupby(
                    'CityTier').size()

                # Convert the result into a DataFrame
                churn_percentage_CT_df = churn_percentage_CT.reset_index()
                churn_percentage_CT_df['CityTier'] = churn_percentage_CT_df['CityTier'].apply(lambda x: 'Tier ' + str(x))
                # Rename columns for clarity
                churn_percentage_CT_df.columns = ['CityTier', 'churn_percentage']
                churn_percentage_CT_df = churn_percentage_CT_df.sort_values('churn_percentage', ascending=True)
                # Plot using Plotly Express

                fig = px.bar(churn_percentage_CT_df, x='churn_percentage', y='CityTier',
                             title='Churn Rate by CityTier',
                             color=churn_percentage_CT_df['churn_percentage'], color_continuous_scale='blues',
                             color_continuous_midpoint=churn_percentage_ppm_df['churn_percentage'].median(), height=400,
                             width=1000)  # Use the median churn percentage as the color midpoint
                fig.update_traces(text=churn_percentage_CT_df['churn_percentage'], texttemplate='%{text:.2%}',
                                  textposition='inside')
                fig.update_yaxes(tickformat=',.0%')
                fig.update_layout(yaxis_title='', xaxis_title='', coloraxis_showscale=False,
                                  plot_bgcolor='rgb(255, 255, 255)', paper_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig, use_container_width=True)

            with col7:
                churn_percentage = df[df['Churn'] == 1].groupby('PreferredLoginDevice').size() / df.groupby(
                    'PreferredLoginDevice').size()

                # Convert the result into a DataFrame
                churn_percentage_df = churn_percentage.reset_index()

                # Rename columns for clarity
                churn_percentage_df.columns = ['PreferredLoginDevice', 'churn_percentage']
                churn_percentage_df = churn_percentage_df.sort_values('churn_percentage', ascending=True)
                # Plot using Plotly Express
                fig = px.bar(churn_percentage_df, x='churn_percentage', y='PreferredLoginDevice',
                             title='Churn Rate by Preferred Login Device',
                             color=churn_percentage_df['churn_percentage'], color_continuous_scale='blues',
                             color_continuous_midpoint=churn_percentage_df['churn_percentage'].median(), height=400,
                             width=1000)  # Use the median churn percentage as the color midpoint
                fig.update_traces(text=churn_percentage_df['churn_percentage'], texttemplate='%{text:.2%}',
                                  textposition='inside')
                fig.update_yaxes(tickformat=',.0%')
                fig.update_layout(yaxis_title='', xaxis_title='', coloraxis_showscale=False,
                                  plot_bgcolor='rgb(255, 255, 255)', paper_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig, use_container_width=True)

            with col8:
                churn_percentage_gender = df[df['Churn'] == 1].groupby('Gender').size() / df.groupby('Gender').size()

                # Convert the result into a DataFrame
                churn_percentage_gender_df = churn_percentage_gender.reset_index()

                # Rename columns for clarity
                churn_percentage_gender_df.columns = ['Gender', 'churn_percentage']
                colors = ['lightblue', 'royalblue']  # Define custom colors for the two genders

                # Plot the pie chart with custom colors
                fig = go.Figure(data=[go.Pie(labels=churn_percentage_gender_df['Gender'],
                                             values=churn_percentage_gender_df['churn_percentage'],
                                             hole=0.5,
                                             marker=dict(colors=colors))])

                fig.update_layout(title_text='Churn Rate by Gender',
                                  plot_bgcolor='rgb(255, 255, 255)',  # Set the background color to white
                                  paper_bgcolor='rgb(255, 255, 255)',  # Set the background color of the paper to white
                                  showlegend=True, height=400, width=1000)
                st.plotly_chart(fig, use_container_width=True)

            with col9:
                total_satisfaction_score = df['CustomerID'].groupby(df['SatisfactionScore']).count()

                # Convert the result into a DataFrame
                total_satisfaction_score_df = total_satisfaction_score.reset_index()

                # Rename columns for clarity
                total_satisfaction_score_df.columns = ['SatisfactionScore', 'total_customers']
                colors = ['#87CEFA', '#00BFFF', '#1E90FF', '#4169E1', '#7B68EE']  # Define custom colors for the two genders

                # Plot the pie chart with custom colors
                fig = go.Figure(data=[go.Pie(labels=total_satisfaction_score_df['SatisfactionScore'],
                                             values=total_satisfaction_score_df['total_customers'],
                                             hole=0.5,
                                             marker=dict(colors=colors))])

                fig.update_layout(title_text='Customer Distribution by Satisfaction Score',
                                  plot_bgcolor='rgb(255, 255, 255)',  # Set the background color to white
                                  paper_bgcolor='rgb(255, 255, 255)',  # Set the background color of the paper to white
                                  showlegend=True, height=400, width=1000)
                st.plotly_chart(fig, use_container_width=True)

            with col10:
                churn_percentage_ppm = df.groupby('PreferredPaymentMode')['CouponUsed'].sum()
                # Convert the result into a DataFrame
                churn_percentage_ppm_df = churn_percentage_ppm.reset_index()

                # Rename columns for clarity
                churn_percentage_ppm_df.columns = ['PreferredPaymentMode', 'CouponUsed']
                churn_percentage_ppm_df = churn_percentage_ppm_df.sort_values('CouponUsed', ascending=True)
                # Plot using Plotly Express

                fig = px.bar(churn_percentage_ppm_df, x='CouponUsed', y='PreferredPaymentMode',
                             title='Coupon Usage by Preferred Payment Mode',
                             color=churn_percentage_ppm_df['CouponUsed'], color_continuous_scale='blues',
                             color_continuous_midpoint=churn_percentage_ppm_df['CouponUsed'].median(), height=400,
                             width=1000)  # Use the median churn percentage as the color midpoint
                fig.update_traces(text=churn_percentage_ppm_df['CouponUsed'], textposition='inside')
                # fig.update_yaxes(tickformat=',.0%')
                fig.update_layout(yaxis_title='', xaxis_title='', coloraxis_showscale=False,
                                  plot_bgcolor='rgb(255, 255, 255)', paper_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig, use_container_width=True)
                # GRAPH--3



        elif st.session_state.page == 'ml_model':

            st.markdown("<h2>Customer Churn Prediction</h2>", unsafe_allow_html=True)
            with st.form("ml-input-form"):
                tenure = st.number_input("Tenure (in months)", 0.0, df.Tenure.max())
                preferredlogindevice = st.selectbox("Choose login device", df1.PreferredLoginDevice.unique().tolist())
                citytier = st.number_input("Enter CityTier", 0, df.CityTier.max())
                warehousetohomedistance = st.slider("WarehouseToHomeDistance", 0.0, df.WarehouseToHome.max())
                preferredpaymentmode = st.selectbox("Choose Preferred Payment Mode", df1.PreferredPaymentMode.unique().tolist())
                gender = st.selectbox('Gender', df1.Gender.unique().tolist())
                hoursspendonapp = st.number_input("HoursSpendOnApp", 0.0, df['HourSpendOnApp'].max())
                noofdeviceregistered = st.number_input('NumberOfDeviceRegistered', 0.0, df['NumberOfDeviceRegistered'].max())
                preferedordercat = st.selectbox('PreferedOrderCat', df1.PreferedOrderCat.unique().tolist())
                SatisfactionScore = st.number_input('SatisfactionScore', 0, df['SatisfactionScore'].max())
                MaritalStatus = st.selectbox('MaritalStatus', df1['MaritalStatus'].unique().tolist())
                NumberOfAddress = st.number_input('NumberOfAddress', 0, df['NumberOfAddress'].max())
                Complain = st.selectbox('Complain', df1['Complain'].unique().tolist())
                OrderAmountHikeFromlastYear = st.slider("OrderAmountHikeFromlastYear", 0.0, df.OrderAmountHikeFromlastYear.max())
                CouponUsed = st.number_input("CouponUsed", 0.0, df.CouponUsed.max())
                OrderCount = st.number_input("OrderCount", 0.0, df.OrderCount.max())
                DaySinceLastOrder = st.number_input("DaySinceLastOrder", 0.0, df.DaySinceLastOrder.max())
                CashbackAmount = st.slider("CashbackAmount", 0.0, df.CashbackAmount.max())

                submit = st.form_submit_button("Predict")
                if submit:
                    prediction = model.predict([[tenure, preferredlogindevice, citytier, warehousetohomedistance,
                                                 preferredpaymentmode, gender, hoursspendonapp,
                                                 noofdeviceregistered, preferedordercat, SatisfactionScore, MaritalStatus,
                                                 NumberOfAddress, Complain, OrderAmountHikeFromlastYear,
                                                 CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount]])[0]
                    if prediction == 1:
                        st.error('Customer Churned')
                    else:
                        st.success('Customer Survived')
