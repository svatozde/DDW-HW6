
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.ticker import FormatStrFormatter



ticks = [0.0,0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
y_ticks = [0.001*i for i in range(1000)]

colours =  ['r','g','b','y']
shapes = ['+','o','x']
print(colours)

with open('pickles/super_result.pickle', 'rb') as handle:
    precisions_map = pickle.load(handle)

    print(precisions_map)
    values = precisions_map[(0.33, 0.33, 0.34)]

    series = []
    labels = []
    for i,k in enumerate(values):
        tmp = []
        labels.append(k)
        for v in values[k]:
            tmp.append(v[2])
        series.append(tmp)


    fig, ax = plt.subplots(figsize=(10,15))
        #fig.subplots_adjust(bottom = 10,  top =20  ,wspace = 50 ,hspace = 50
    for i,ser in enumerate(series):

        ax.plot(ticks, ser, str(colours[i%len(colours)])+str(shapes[i%3])+'-' ,label=str(labels[i]))

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ticks)
    ax.set_xticks([0.5*a for  a in range(len(ticks))])
    ax.set_title('Precision for different counts of users used guess rating')
    ax.legend(loc='lower left')
    ax.grid(axis='both')
    plt.savefig('figs/recall_comparsion.jpg')
    plt.clf()
    plt.cla()
