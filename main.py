import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# NN
from sklearn.neighbors import NearestNeighbors


# For all of the methods we have the same logic to define the clusters
# Returns the kmeans object, x, real y and the predicted cluster for the majority class

def run_kmeans(k, df, maj):
    df = df[df['y'] == maj]
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    kmeans = KMeans(init="k-means++", n_clusters=k, random_state=0).fit(x_train)
    y_pred = kmeans.predict(x)
    return kmeans, x, y, y_pred


# Helper function to define the majority and minority class as well as its size

def maj_min(df):
    values = df['y'].unique()
    size = []
    for value in values:
        size.append(df[df['y'] == value].shape[0])
    if(size[0] > size[1]):
        return values[0],size[0],values[1],size[1]
    else:
        return values[1],size[1],values[0],size[0]

# Helper function to calculate all the performance metrics

def performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, auc

# Model function for the SVM to predict based on X_train, y_train
# Inside this function a pipeline is made where the input is scaled.

def predict_svm(X_train, y_train, X_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

# Another Model function, this time for the Logistic Regression Classifier, based on the same input parameters.

def predict_lr(X_train, y_train, X_test):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf.predict(X_test)

# The first method: Find the same amount of clusters in the majority class as we have minority class data points
# and combine these centroids with the minority class data points

def method1(df, maj, min, min_n):
    kmeans, _, _, _ = run_kmeans(min_n, df, maj)
    minority = df[df['y']==min]
    majority = pd.DataFrame(kmeans.cluster_centers_)
    majority['y'] = maj
    majority.columns = minority.columns
    result = pd.concat([minority,majority], axis=0)
    return result

# The second method: Create k clusters and get x random samples from every cluster so that k * x is the amount of
# data points in the minority class

def method2(df, maj, min, min_n, k):
    _, x, y, y_pred = run_kmeans(k, df, maj)
    x = x.reset_index(drop=True)
    y_pred = pd.DataFrame(y_pred, columns=['y'])
    y_pred = y_pred.reset_index(drop=True)
    dataset = pd.concat([x,y_pred], axis=1)
    clusters, cluster_sizes = np.unique(y_pred, return_counts=True)
    size_to_find = round(min_n / k)
    majority = pd.DataFrame()
    for i, cluster in enumerate(cluster_sizes):
        find_n = size_to_find
        if(cluster <= find_n):
            majority = pd.concat([dataset[dataset['y'] == clusters[i]].iloc[:,:-1],majority], axis=0)
            find_n = find_n - cluster
            # Find the remaining
            for j, cluster_remaining in enumerate(clusters):
                if(cluster_remaining >= find_n):
                    majority = pd.concat([dataset[dataset['y'] == clusters[i]].sample(find_n).iloc[:,:-1], majority], axis=0)
                    break
        else:
            majority = pd.concat([dataset[dataset['y'] == clusters[i]].sample(find_n).iloc[:,:-1], majority], axis=0)
    majority['y'] = maj
    minority = df[df['y']==min]
    result = pd.concat([minority,majority], axis=0)
    return result

# The third method: Find the same amount of clusters in the majority class as we have minority class data points
# and combine these Top1 neighbours of these centroids with the minority class data points

def method3(df, maj, min, min_n):
    kmeans, x, y, _ = run_kmeans(min_n, df, maj)
    minority = df[df['y']==min]
    majorities = df[df['y']==maj].iloc[:,:-1]
    centers = kmeans.cluster_centers_
    neighbours = NearestNeighbors(n_neighbors=1)
    neighbours.fit(majorities)
    majority = pd.DataFrame()
    for center in centers:
        majority = pd.concat([majority,pd.DataFrame(majorities.iloc[neighbours.kneighbors([center])[1][0]])], axis=0)
    majority['y'] = maj
    majority.columns = minority.columns
    result = pd.concat([minority,majority], axis=0)
    return result

# The fourth method: Find the same amount of clusters in the majority class as we have minority class data points
# and combine these TopN neighbours of these centroids with the minority class data points so that N * k is the same as
# the amount of minority class data points

def method4(df, maj, min, min_n, k):
    kmeans, x, y, _ = run_kmeans(k, df, maj)
    minority = df[df['y']==min]
    majorities = df[df['y']==maj].iloc[:,:-1]
    centers = kmeans.cluster_centers_
    neighbours_n = int(round(min_n/k,0))
    neighbours = NearestNeighbors(n_neighbors=neighbours_n)
    neighbours.fit(majorities)
    majority = pd.DataFrame()
    for center in centers:
        for neighbour in neighbours.kneighbors([center])[1]:
            majority = pd.concat([majority,pd.DataFrame(majorities.iloc[neighbour])], axis=0)
    majority['y'] = maj
    majority.columns = minority.columns
    result = pd.concat([minority,majority], axis=0)
    return result

st.write("Data Science - Wessel van de Goor")
st.title('Sliders')
dataset = st.radio("Select a dataset",('datasetPoker86', 'datasetPoker97'))

df = pd.read_csv(f'{dataset}.csv', delimiter=';')

st.write(f'Selected Dataset: {dataset}')
st.write('Quick view of the data')
st.write(df.head(5))
st.write('Data Shape')
st.write(df.shape)
maj, maj_n, min, min_n = maj_min(df)
st.write(f"The Majority - Minority Ratio is: {round(maj_n/min_n,1)}:1")

model_name = st.radio("Model",('SVM','LogisticRegression'))
methods = ('1. K-means++ with centroids', '2. K-means++ with random sampling',
           '3. K-means++ with Top1 Neighbours', '4. K-means++ with TopN Neighbours',
           '0. No under-sampling')
method = st.radio("Method",methods)
if(method == methods[1] or method == methods[3]):
    k = st.slider("Amount of clusters", 1, 20)

# Check what method has been selected
train_df = pd.DataFrame()
if(method == methods[0]):
    train_df = method1(df,maj, min, min_n)
elif(method == methods[1]):
    train_df = method2(df,maj, min, min_n, k)
elif(method == methods[2]):
    train_df = method3(df,maj, min, min_n)
elif(method == methods[3]):
    train_df = method4(df,maj, min, min_n, k)
elif(method == methods[4]):
    train_df = df.iloc[int(round(df.shape[0]*0.2,0)):,:]

# Map classes to numbers
y_true = df.iloc[:,-1]
y_true = np.where(y_true == "positive",1,0)
y_pred = df.iloc[:,-1]

# Check what algorithm has been chosen
if(model_name == 'SVM'):
    y_pred = predict_svm(train_df.iloc[:, :-1],train_df.iloc[:,-1],df.iloc[:,:-1])
elif(model_name == 'LogisticRegression'):
    y_pred = predict_lr(train_df.iloc[:, :-1],train_df.iloc[:,-1],df.iloc[:,:-1])
y_pred = np.where(y_pred == "positive",1,0)

st.title('Balanced Dataset')
st.write(train_df)

accuracy, precision, recall, f1, auc = performance(y_true, y_pred)
st.title('Performance')
st.write(f"Accuracy = {accuracy}")
st.write(f"Precision = {precision}")
st.write(f"Recall = {recall}")
st.write(f"F-Score = {f1}")
st.write(f"AUC = {auc}")