import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
st.set_option('deprecation.showPyplotGlobalUse', False)

# title
st.title('Customer Churn')
background_im = cv2.imread('customer_churn.jpeg')


df_prep = pd.read_csv('data_preprocessed.csv')
df_raw = pd.read_csv('data_raw.csv')

X = df_prep.drop(['churn_flag', 'Unnamed: 0'], axis=1)
y = df_prep['churn_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_models = {}
sm_models = {}
base_names = ['Logistic Regression', 'Random Forest', 'SVC', 'XGB', 'KNN', 'Naive bayes']
sm_names = ['Logistic Regression SM', 'Random Forest SM', 'SVC SM', 'XGB SM', 'KNN SM', 'Naive bayes SM']
for base_model, base_name, sm_model, sm_name in zip(['lr', 'rf', 'svm', 'xgb', 'knn', 'nb'],base_names,
                                ['lr_sm', 'rf_sm', 'svm_sm', 'xgb_sm', 'knn_sm', 'nb_sm'],sm_names):

    base_models[base_name] = pickle.load(open('models/' + base_model + '.pkl', 'rb'))
    sm_models[sm_name] = pickle.load(open('models/' + sm_model + '.pkl', 'rb'))


def main():
    activities = ['Main', 'Visualization and EDA', 'Model Selection and Performance', 'Feature Importances', 'About']
    option = st.sidebar.selectbox('Select Option', activities)
    if option in activities:
        if option == 'Main':
            st.image(background_im, use_column_width=True)
            st.subheader('General info')
            st.info('Visualisation and EDA section contains some plots and graphs as well as some basic '
                     'information based on the raw and preprocessed data.')
            st.info('Model Selection and Performance section provides information about tested models and their relative performance.')
            st.info('In the Feature Importances section importance of the top 30 features predicted by different models is shown.')

        if option == 'Visualization and EDA':
            st.subheader('Visualization and EDA')
            selected_dataset = st.sidebar.selectbox('Selected dataset', ['Raw', 'Preprocessed'])
            if selected_dataset == 'Raw':
                if st.checkbox('Display shape'):
                    st.write('Size of the raw data: ', df_raw.shape)
                if st.checkbox('Display summary'):
                    st.write(df_raw.describe())
                if st.checkbox('Display null values'):
                    st.write(df_raw.isnull().sum())
                if st.checkbox('Display data types'):
                    st.write(df_raw.dtypes)
                if st.checkbox('Select Multiple columns to plot (max 5)'):
                    selected_columns = st.multiselect('Select your preferred columns', df_raw.columns)
                    if len(selected_columns) <= 5:
                        df = df_raw[selected_columns[:5]]
                    else:
                        st.warning('You have selected too many columns. Reduce the number.')
                if st.checkbox('Display heatmap'):
                    fig, ax = plt.subplots()
                    sns.heatmap(df.corr(), annot=True, ax=ax)
                    st.pyplot(fig)
                if st.checkbox('Display pairplot'):
                    st.write(sns.pairplot(df, diag_kind='kde'))
                    st.pyplot()

            elif selected_dataset == 'Preprocessed':
                if st.checkbox('Display shape'):
                    st.write('Size of the preprocessed data: ', df_prep.shape)
                if st.checkbox('Display summary for preprocessed data'):
                    st.write(df_prep.describe())
                if st.checkbox('Display top N correlations with target class in the preprocessed data'):
                    N = st.slider('Number of features', 0, len(df_prep.columns))
                    fig, ax = plt.subplots()
                    df_prep.corr()['churn_flag'].sort_values(ascending=False)[:N].plot(ax=ax, kind ='bar')
                    st.pyplot(fig)
                if st.checkbox('Select Multiple columns to plot (max 5)'):
                    selected_columns = st.multiselect('Select your preferred columns', df_prep.columns)
                    if len(selected_columns) <= 5:
                        df = df_prep[selected_columns[:5]]
                    else:
                        st.warning('You have selected too many columns. Reduce the number.')
                if st.checkbox('Display heatmap'):
                    fig, ax = plt.subplots()
                    sns.heatmap(df.corr(), annot=True, ax=ax)
                    st.pyplot(fig)
                if st.checkbox('Display pairplot'):
                    st.write(sns.pairplot(df, diag_kind='kde'))
                    st.pyplot()

        if option == 'Model Selection and Performance':
            st.subheader('Model Selection and Performance')
            selected_sampling_type = st.sidebar.selectbox('Select data sampling type', ['No sampling', 'SMOTEENN'])
            if selected_sampling_type == 'No sampling':
                selected_model = st.sidebar.selectbox('Select Model', ['Logistic Regression', 'Random Forest',
                                                                       'SVC', 'XGB', 'KNN', 'Naive bayes', 'All models comparison'])
                if selected_model == 'All models comparison':
                    st.info('ROC Curve')
                    fig, ax = plt.subplots()
                    for name, model in base_models.items():
                        plot_roc_curve(model, X_test, y_test, ax=ax)
                    st.pyplot(fig)
                else:
                    model = base_models[selected_model]

            elif selected_sampling_type == 'SMOTEENN':
                selected_model = st.sidebar.selectbox('Select Model', ['Logistic Regression SM', 'Random Forest SM',
                                                                        'SVC SM', 'XGB SM', 'KNN SM',
                                                                       'Naive bayes SM', 'All models comparison'])
                if selected_model == 'All models comparison':
                    st.info('ROC Curve')
                    fig, ax = plt.subplots()
                    for name, model in sm_models.items():
                        plot_roc_curve(model, X_test, y_test, ax=ax)
                    st.pyplot(fig)
                else:
                    model = sm_models[selected_model]

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

        if option == 'Feature Importances':
            st.subheader('Feature Importances')
            importances_xgb = pd.DataFrame({
                'Feature': X.columns,
                'Importance': sm_models['XGB SM'].feature_importances_
            })
            importances_xgb = importances_xgb.sort_values(by='Importance', ascending=False)
            importances_xgb = importances_xgb.set_index('Feature')

            importances_rf = pd.DataFrame({
                'Feature': X.columns,
                'Importance': sm_models['Random Forest SM'].feature_importances_
            })
            importances_rf = importances_rf.sort_values(by='Importance', ascending=False)
            importances_rf = importances_rf.set_index('Feature')

            fig, ax = plt.subplots()
            importances_xgb[:30].plot(kind='bar', title='Feature Importances XGB Classifier', ax=ax)
            st.pyplot(fig)


            fig1, ax1 = plt.subplots()
            importances_rf[:30].plot(kind='bar', title='Feature Importances Random Forest', ax=ax1)
            st.pyplot(fig1)


        if option == 'About':
            st.subheader('About')
            st.write('This is an interactive website for the Customer Churn ML Project. Data was taken from Udemy ML course.')



if __name__ == '__main__':
    main()