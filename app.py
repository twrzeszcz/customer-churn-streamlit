import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import gc
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from yellowbrick.classifier import ClassificationReport
st.set_option('deprecation.showPyplotGlobalUse', False)

gc.enable()


def load_base_models(model_name):
    base_names = ['Logistic Regression', 'Random Forest', 'SVC', 'XGB', 'KNN', 'Naive bayes']
    base_names_models = ['lr', 'rf', 'svm', 'xgb', 'knn', 'nb']
    names = dict(zip(base_names, base_names_models))
    model = pickle.load(open('models/' + names[model_name] + '.pkl', 'rb'))

    return model

def load_sm_models(model_name):
    sm_names = ['Logistic Regression SM', 'Random Forest SM', 'SVC SM', 'XGB SM', 'KNN SM', 'Naive bayes SM']
    sm_names_models = ['lr_sm', 'rf_sm', 'svm_sm', 'xgb_sm', 'knn_sm', 'nb_sm']
    names = dict(zip(sm_names, sm_names_models))
    model = pickle.load(open('models/' + names[model_name] + '.pkl', 'rb'))

    return model


def load_data_raw():
    df_raw = pd.read_csv('data_raw.csv')
    return df_raw


def load_data_prep():
    df_prep = pd.read_csv('data_preprocessed.csv')
    df_prep.drop('Unnamed: 0', axis=1, inplace=True)
    return df_prep

def preprocess(df_prep):
    X = df_prep.drop(['churn_flag'], axis=1)
    y = df_prep['churn_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    del X_train, y_train, y, scaler

    return X, X_test, y_test

def main_section():
    st.title('Customer Churn Project')
    background_im = cv2.imread('images/customer_churn.jpeg')
    st.image(background_im, use_column_width=True)
    st.subheader('General info')
    st.info('Visualisation and EDA section contains some plots and graphs as well as some basic '
            'information based on the raw and preprocessed data.')
    st.info('Model Selection and Performance section provides information about tested models and their relative performance.')
    st.info('In the Feature Importances section importance of the top 30 features predicted by different models is shown.')
    del background_im
    gc.collect()

def vis_and_eda():
    st.title('Visualization and EDA')
    selected_dataset = st.sidebar.selectbox('Selected dataset', ['Raw', 'Preprocessed'])
    if selected_dataset == 'Raw':
        df_raw = load_data_raw()
        st.success('Data successfully loaded')
        if st.checkbox('Display shape'):
            st.write('Size of the raw data: ', df_raw.shape)
        if st.checkbox('Display summary'):
            st.write(df_raw.describe())
        if st.checkbox('Display null values'):
            st.write(df_raw.isnull().sum())
        if st.checkbox('Display data types'):
            st.write(df_raw.dtypes)
        if st.checkbox('Select Multiple columns to plot'):
            selected_columns = st.multiselect('Select your preferred columns', df_raw.columns)
        if st.checkbox('Display heatmap'):
            fig, ax = plt.subplots()
            sns.heatmap(df_raw[selected_columns].corr(), annot=True, ax=ax)
            st.pyplot(fig)
        if st.checkbox('Display pairplot'):
            st.write(sns.pairplot(df_raw[selected_columns], diag_kind='kde'))
            st.pyplot()
        del df_raw
        gc.collect()

    elif selected_dataset == 'Preprocessed':
        df_prep = load_data_prep()
        st.success('Data successfully loaded')
        if st.checkbox('Display shape'):
            st.write('Size of the preprocessed data: ', df_prep.shape)
        if st.checkbox('Display summary for preprocessed data'):
            st.write(df_prep.describe())
        if st.checkbox('Display top N correlations with target class in the preprocessed data'):
            N = st.slider('Number of features', 0, len(df_prep.columns))
            fig, ax = plt.subplots()
            df_prep.corr()['churn_flag'].sort_values(ascending=False)[:N].plot(ax=ax, kind ='bar')
            st.pyplot(fig)
        if st.checkbox('Select Multiple columns to plot'):
            selected_columns = st.multiselect('Select your preferred columns', df_prep.columns)
        if st.checkbox('Display heatmap'):
            fig, ax = plt.subplots()
            sns.heatmap(df_prep[selected_columns].corr(), annot=True, ax=ax)
            st.pyplot(fig)
        if st.checkbox('Display pairplot'):
            st.write(sns.pairplot(df_prep[selected_columns], diag_kind='kde'))
            st.pyplot()
        del df_prep
        gc.collect()

def model_selection_and_performance():
    st.title('Model Selection and Performance')
    selected_sampling_type = st.sidebar.selectbox('Select data sampling type', ['No sampling', 'SMOTEENN'])
    df_prep = load_data_prep()
    X, X_test, y_test = preprocess(df_prep)
    del X

    if selected_sampling_type == 'No sampling':
        selected_model = st.sidebar.selectbox('Select Model', ['Logistic Regression', 'Random Forest',
                                                               'SVC', 'XGB', 'KNN', 'Naive bayes',
                                                               'All models comparison'])
        if selected_model == 'All models comparison':
            st.info('ROC Curves comparison')
            roc_all = cv2.imread('images/base_models_comparison.jpg')
            st.image(roc_all, use_column_width=True)
            del roc_all
        else:
            model = load_base_models(selected_model)

        gc.collect()

    elif selected_sampling_type == 'SMOTEENN':
        selected_model = st.sidebar.selectbox('Select Model', ['Logistic Regression SM', 'Random Forest SM',
                                                               'SVC SM', 'XGB SM', 'KNN SM',
                                                               'Naive bayes SM', 'All models comparison'])
        if selected_model == 'All models comparison':
            st.info('ROC Curves comparison')
            roc_all = cv2.imread('images/sm_models_comparison.jpg')
            st.image(roc_all, use_column_width=True)
            del roc_all
        else:
            model = load_sm_models(selected_model)

        gc.collect()


    if selected_model != 'All models comparison':
        fig, ax = plt.subplots()
        visualizer = ClassificationReport(model, classes=['non-churn', 'churn'], support=True, ax=ax)
        visualizer.score(X_test, y_test)
        visualizer.show()
        st.pyplot(fig)
        st.info('Confusion Matrix')
        fig1, ax1 = plt.subplots()
        plot_confusion_matrix(model, X_test, y_test, ax=ax1)
        st.pyplot(fig1)
        st.info('ROC Curve')
        fig2, ax2 = plt.subplots()
        plot_roc_curve(model, X_test, y_test, ax=ax2)
        st.pyplot(fig2)
        del X_test, y_test, fig, ax, fig1, ax1, fig2, ax2, model
        gc.collect()

def feature_importances():
        st.title('Feature Importances')
        df_prep = load_data_prep()
        X, X_test, y_test = preprocess(df_prep)
        del X_test, y_test

        xgb = load_sm_models('XGB SM')
        rf = load_sm_models('Random Forest SM')

        importances_xgb = pd.DataFrame({
            'Feature': X.columns,
            'Importance': xgb.feature_importances_
        })
        importances_xgb = importances_xgb.sort_values(by='Importance', ascending=False).set_index('Feature')

        importances_rf = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        })
        importances_rf = importances_rf.sort_values(by='Importance', ascending=False).set_index('Feature')

        fig, ax = plt.subplots()
        importances_xgb[:30].plot(kind='bar', title='Feature Importances XGB Classifier', ax=ax)
        st.pyplot(fig)

        fig1, ax1 = plt.subplots()
        importances_rf[:30].plot(kind='bar', title='Feature Importances Random Forest', ax=ax1)
        st.pyplot(fig1)
        del fig, ax, importances_xgb, fig1, ax1, importances_rf, X, xgb, rf
        gc.collect()

activities = ['Main', 'Visualization and EDA', 'Model Selection and Performance', 'Feature Importances', 'About']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Visualization and EDA':
    vis_and_eda()
    gc.collect()

if option == 'Model Selection and Performance':
    model_selection_and_performance()
    gc.collect()

if option == 'Feature Importances':
    feature_importances()
    gc.collect()

if option == 'About':
    st.title('About')
    st.write('This is an interactive website for the Customer Churn ML Project. Data was taken from Udemy ML course.')


