"""Different functions for visualizations
"""
import seaborn as sn
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(
    net,
    testloader,
    annotations_file,
    classes_name='category',
    fig_name='confusion_matrix.png'
    ):
    y_pred = []
    y_true = []

    # iterate over test data
    for sample in testloader:
        inputs = sample['image']
        labels = sample['label']
        inputs = inputs.to(device)

        output = net(inputs) # Feed Network

        output = (torch.max(output, 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    df_an = pd.read_csv(annotations_file)
    classes = df_an[[classes_name, 'label']].drop_duplicates().sort_values(by='label')[classes_name].tolist()
    print(classes)
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    print(np.sum(cf_matrix))
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=[i for i in classes],
                        # columns=[i for i in classes])
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix))

    plt.figure(figsize=(24, 14))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(fig_name)
    print((np.array(y_true) == np.array(y_pred)).sum() / len(y_true))
