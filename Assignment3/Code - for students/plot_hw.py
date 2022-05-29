from turtle import color
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import show
import numpy as np
from bokeh.io import output_notebook
# output_notebook()

def plot_trans(train_los):
    no_epochs = len(train_los)
    p = figure(x_axis_label='Number of epochs',y_axis_label='Train Loss', width=550,height=550,x_range = (0,no_epochs), title='Train loss')
    p.line(np.arange(len(train_los)), train_los, legend_label="Train Loss")
    return p

# plot train & loss

def plt_trn_test(train_los,test_los):
    no_epochs = len(train_los)
    p = figure(x_axis_label='Number of epochs',y_axis_label='Train Loss', width=550,height=550,x_range = (0,no_epochs), title='Cross Entropy')    
    p.line(np.arange(len(train_los)), train_los, legend_label="Test Loss",color='black')
    p.line(np.arange(len(test_los)), test_los, legend_label="Test Loss",color='blue')
    p.title('Train loss and Test loss of each epoch')
    return p

