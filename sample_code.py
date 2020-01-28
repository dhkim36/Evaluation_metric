import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ComputePrecision_Recall(data, total_object):
    graph = pd.DataFrame(columns=['precision', 'recall'])
    sort_data = data.sort_values(by='confidence', ascending=False)
    sort_data = sort_data.reset_index(drop=True)
    tp = 0

    for i in range(0, data.shape[0]):
        if sort_data['measure'][i] != 'FP':
            tp += 1
        precision = tp / (i+1)
        recall = tp / total_object
        append_data = pd.Series([precision, recall], index = ['precision', 'recall'])
        graph = graph.append(append_data, ignore_index=True)

    return precision, recall, graph

def PrecRecCurve(graph):
        graph.plot.line(x="recall", y="precision")

def ComputeAveragePrecision_11P(graph):
    graph = graph.to_numpy()
    recall_eleven = np.arange(0, 1.1, 0.1)
    graph_eleven = np.array([])

    recall_eleven = np.round(recall_eleven, 5)
    for i in recall_eleven :
        try:
            graph_eleven = np.append(graph_eleven, np.max(graph[:, 0][((i+0.1) > graph[:,1]) & (graph[:,1] >= i)]))
        except:
            graph_eleven = np.append(graph_eleven, 0)

    for i, j in enumerate(graph_eleven):
        graph_eleven[i] = np.max(graph_eleven[i:])

    print('\n11_interpolation\n', np.sum(graph_eleven)/11)

    #graph.plot.line(x="recall", y="precision")
    #plt.show()

def ComputeAveragePrecision_every(graph):
    graph = graph.to_numpy()
    recall_every, idx = np.unique(graph[:,1], return_index=True)
    sum = 0
    last_graph = 0
    for i in graph:
        max_arg = np.argmax(graph[:, 0])
        sum += graph[max_arg,0] * (graph[max_arg,1] - last_graph)
        last_graph = graph[max_arg,1]
        graph[:max_arg+1] = 0
    print('\nevery interpolation\n', sum)



# To Detect Num of Object
#csv_path = input('csv path : ')
csv_path = 'results_3.csv'
NUM_OF_OBJECT = 5
#NUM_OF_OBJECT = int(input('total object input : '))
data_path = os.path.join(os.getcwd(), csv_path)
data = pd.read_csv(data_path)

precision, recall, graph = ComputePrecision_Recall(data, NUM_OF_OBJECT)
PrecRecCurve(graph)
ComputeAveragePrecision_11P(graph)
ComputeAveragePrecision_every(graph)
