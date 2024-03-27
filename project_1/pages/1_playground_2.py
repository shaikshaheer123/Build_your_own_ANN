import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras

from keras.models import Sequential
from keras.layers import InputLayer, Dense

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from keras.optimizers import SGD

import mlxtend
from mlxtend.plotting import plot_decision_regions


with st.sidebar:
    st.header('Tensorflow Playground')
    # Creating Random Data set-----------------------------------------------------------------------------------------------------------------
    st.subheader('Select Data Set')
    u_shape = pd.read_csv('data/1.ushape.csv')
    concentric_circle_1 = pd.read_csv('data/2.concerticcir1.csv')
    concentric_circle_2 = pd.read_csv('data/3.concertriccir2.csv')
    linearly_Separable = pd.read_csv('data/4.linearsep.csv')
    outlier = pd.read_csv('data/5.outlier.csv')
    overlap = pd.read_csv('data/6.overlap.csv')
    xor = pd.read_csv('data/7.xor.csv')
    two_spiral = pd.read_csv('data/8.twospirals.csv')

    data_set = st.selectbox("Select Data Set:", 
                            ("U - Shape", "Concentric Circle 1", 'Concentric Circle 2', 'Linearly Separable',
                             'Outlier', 'Overlap', 'XOR', 'Two Spiral'), 
                            placeholder="Select Data Set...")
    if data_set=="U - Shape":
        data = u_shape
        n_samples, col = data.shape
    elif data_set=="Concentric Circle 1":
        data = concentric_circle_1
        n_samples, col = data.shape
    elif data_set=="Concentric Circle 2":
        data = concentric_circle_2
        n_samples, col = data.shape
    elif data_set=="Linearly Separable":
        data = linearly_Separable
        n_samples, col = data.shape
    elif data_set=="Outlier":
        data = outlier
        n_samples, col = data.shape
    elif data_set=="Overlap":
        data = overlap
        n_samples, col = data.shape
    elif data_set=="XOR":
        data = xor
        n_samples, col = data.shape
    else:
        data = two_spiral
        n_samples, col = data.shape

    data.columns = ['F1', 'F2', 'Y']
    fv = data.iloc[:, :2]
    cv = data.iloc[:, -1]

st.header('Results')    
fig_data, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
sns.scatterplot(x=fv.iloc[:, 0], y=fv.iloc[:, 1], hue=cv, ax=ax)

with st.container():
    col1, col2, col3 = st.columns([3,6,3])
    with col2:
        st.subheader("Data")
        st.pyplot(fig=fig_data)

with st.sidebar:
    # Train Test split---------------------------------------------------------------------------------------------------------------------------------------------------- 
    st.subheader('Train Test Split')
    test_size = st.slider('Enter Test Size:', min_value=10, max_value=90, value=20, step=5)
    random_state_train_test_split = st.number_input('Randomstate - Train Test Split:', min_value=0, max_value=100000, value="min", placeholder='Enter any integer...')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(fv, cv, test_size=test_size, stratify=cv, random_state=random_state_train_test_split)
    
    std=StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)

    # Creating Neural Network--------------------------------------------------------------------------------------------------------------------
    st.subheader('Creating Neural Network')
    hidden_layers = st.number_input('Enter No.of Hidden Layers:', min_value=0, max_value=3, value="min", placeholder=1)
    activation_function_hiddenlayers = st.selectbox("Activation Function - Hidden Layers:", 
                                                    ("sigmoid", "softmax", "tanh"), index=None, placeholder="Select Activation Function...")
    
    bias_hidden_layers = st.selectbox("Use_Bias - Hidden Layers:", 
                                                    (True, False), index=None, placeholder="Select Bias...")
    
    activation_function_outputlayers = st.selectbox("Activation Function - Output Layers:", 
                                                    ("sigmoid", "softmax", "tanh"), index=None, placeholder="Select Activation Function...")
    bias_output_layer = st.selectbox("Use_Bias - Output Layer:", 
                                                    (True, False), index=None, placeholder="Select Bias...")
    
    no_of_neurons = 2 * hidden_layers
    model=Sequential()
    model.add(InputLayer(input_shape=(2,)))

    for layer in range(hidden_layers):
        model.add(Dense(units=no_of_neurons, activation=activation_function_hiddenlayers, use_bias=bias_hidden_layers))
        no_of_neurons -= 2

    model.add(Dense(1, activation=activation_function_outputlayers, use_bias=bias_output_layer))


    # Compile Model--------------------------------------------------------------------------------------------------------------------------------
    st.subheader('Compile Model')
    loss_function = st.selectbox("Loss Function:", 
                                 ("binary_crossentropy", "sparse_categorical_crossentropy"), placeholder="Select Loss Function...")
    
    evalution_metrics = [st.selectbox("Evaluation Metrics:", 
                                 ("accuracy", "binary_accuracy"), placeholder="Select Evalution Metrics...")]
    learning_rate = st.selectbox("Learning Rate:", 
                                 (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 0.003, 0.03, 0.3), placeholder="Select Evalution Metrics...")
    
    sgd = SGD(learning_rate=float(learning_rate))
    model.compile(optimizer=sgd, loss=loss_function, metrics=evalution_metrics)

    #Training Model-------------------------------------------------------------------------------------------------------------------------------
    st.subheader('Training Model')
    no_of_epochs = st.number_input('Enter No.of Epochs:', min_value=1, max_value=10000, value="min", placeholder=10)
    batch_size = st.number_input('Enter Batch Size:', min_value=1, max_value=n_samples, value="min", placeholder=10)
    validation_split = st.number_input('Enter Validation Split:', min_value=0.1, max_value=0.9, value="min", placeholder=0.2, step=0.05)
    btn_response = st.button('Start Training...')
    if btn_response:
        history = model.fit(X_train, y_train, epochs=no_of_epochs, batch_size=batch_size, validation_split=validation_split)
   

# Page-------------------------------------------------------------------------
fig_data, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
sns.scatterplot(x=fv.iloc[:, 0], y=fv.iloc[:, 1], hue=cv, ax=ax)

y_test_array = y_test.values.astype(np.int_)
print(type(y_test), type(y_test_array))
fig_db, ax1 = plt.subplots(figsize=(8,8), constrained_layout=True)
plot_decision_regions(X_test,y_test_array,clf=model, ax=ax1)


with st.container():
    col1, col2, col3 = st.columns([5,2,5])
    with col1:
        st.subheader("Data")
        st.pyplot(fig=fig_data)
    # with col2:
    #     st.image('data/arrow.png')
    with col3:
        st.subheader('Decision Boundary')
        st.pyplot(fig=fig_db)


with st.container():
    fig_losses, ax_l = plt.subplots(figsize=(5,5), constrained_layout=True)
    sns.lineplot(x=range(1,no_of_epochs + 1),y=history.history["loss"],label="train_loss", ax=ax_l)
    sns.lineplot(x=range(1,no_of_epochs + 1),y=history.history["val_loss"],label="val_loss", ax=ax_l)
    plt.legend()
    plt.show()
    st.pyplot(fig=fig_losses)