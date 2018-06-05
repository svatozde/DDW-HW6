import pickle
import numpy as np
import  matplotlib
import  matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


super_results = {}
with open('pickles/super_correct_matric.pickle', 'rb') as handle:
    super_results =  pickle.load(handle)

    ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    labels = [0.0,0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    colours = ['r', 'g', 'b', 'y']
    shapes = ['+', 'o', 'x']

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(ncols=4, nrows=len(super_results) * 5, figsize=(10, 5))
    fig.tight_layout()
    row = -1

    series = {}

    for key1 in [(0.0,1.0),(1.0,0.0)]:
        if key1 not in series:
            series[key1]={}
        for key2 in super_results[key1]:
            if key2 not in series[key1]:
                series[key1][key2] = {}
                series[key1][key2]['tp'] = []
                series[key1][key2]['tn'] = []
                series[key1][key2]['fp'] = []
                series[key1][key2]['fn'] = []
                series[key1][key2]['pr'] = []
                series[key1][key2]['re'] = []
                series[key1][key2]['ac'] = []
                series[key1][key2]['sum'] = []

            conf_matrix = super_results[key1][key2]

            conf_arr = np.zeros((10,10))

            for i,t1 in enumerate(ticks):
                for j, t2 in enumerate(ticks):
                    if t1 in conf_matrix and t2 in conf_matrix[t1]:
                        conf_arr[i][j] = conf_matrix[t1][t2]


            for cr in range(0,10):
                tn = sum(map(sum,conf_arr[:cr,:cr]))
                #print(conf_arr[:cr,cr:])
                fp = sum(map(sum,conf_arr[:cr,cr:]))
                #print(conf_arr[cr:,:cr])
                fn = sum(map(sum,conf_arr[cr:,:cr]))
                #print(conf_arr[cr:,cr:])
                tp = sum(map(sum,conf_arr[cr:,cr:]))

                series[key1][key2]['tp'].append(tp)
                series[key1][key2]['tn'].append(tn)
                series[key1][key2]['fp'].append(fp)
                series[key1][key2]['fn'].append(fn)
                series[key1][key2]['pr'].append(tp/(tp+fp))
                series[key1][key2]['re'].append(tp/(tp+fn))
                series[key1][key2]['ac'].append((tp+tn)/(tp+tn+fp+fn))
                series[key1][key2]['sum'].append(tp + fn)

        for i in [1,2,3,5,10,50,500]:
            fig, ax = plt.subplots(ncols=4, figsize=(10, 5))

            ax[0].set_title('Precision for ' + str(i) + ' recomandations' )
            ax[1].set_title('Recall for ' + str(i) + ' recomandations')
            ax[2].set_title('Accuracy for ' + str(i) + ' recomandations')
            ax[3].set_title('Count of movies with \n actual rating higher than threshold  \n for ' + str(i) + ' recomandations')
            m = 0
            for k1 in series:
                series[k1][i]

                ax[0].plot(ticks, series[k1][i]['pr'], str(colours[m % len(colours)]) + str(shapes[(m) % len(shapes)]) + '-',
                           label=str(k1))
                ax[1].plot(ticks, series[k1][i]['re'], str(colours[m % len(colours)]) + str(shapes[(m) % len(shapes)]) + '-',
                           label=str(k1))
                ax[2].plot(ticks, series[k1][i]['ac'], str(colours[m % len(colours)]) + str(shapes[(m) % len(shapes)]) + '-',
                           label=str(k1))
                ax[3].plot(ticks, series[k1][i]['sum'],
                           str(colours[m % len(colours)]) + str(shapes[(m) % len(shapes)]) + '-',
                           label=str(k1))
                m+=1
            for a in ax[:3]:
                a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                a.yaxis.set_ticks(np.arange(0.0, 1.1, 0.05))
                a.set_xticklabels(labels)
                a.set_xticks([0.5 * a for a in range(len(labels))])
                a.legend(loc='lower left')
                a.grid(axis='both')

            ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[3].set_xticklabels(labels)
            ax[3].set_xticks([0.5 * a for a in range(len(labels))])
            ax[3].legend(loc='lower left')
            ax[3].grid(axis='both')

            plt.savefig('figs_new/rating_comparsion'+str(i)+'.jpg')
            plt.clf()
            plt.cla()



