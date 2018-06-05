
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

    labels = []
    fig, ax = plt.subplots(ncols=3, figsize=(15, 20))

    ax[0].set_title('Precision')
    ax[1].set_title('Recall')
    ax[2].set_title('Accuracy')

    for i,k in enumerate(precisions_map):
       # for j,kk in  enumerate(precisions_map[k]):
            label = str(k)
            pp = []
            rr = []
            aa = []

            for s,p,r,a in precisions_map[k][300]:
                pp.append(p)
                rr.append(r)
                aa.append(a)

            ax[0].plot(ticks,pp, str(colours[i % len(colours)]) + str(shapes[(i) % len(shapes)]) + '-', label=str(label))
            ax[1].plot(ticks,rr, str(colours[i % len(colours)]) + str(shapes[(i) % len(shapes)]) + '-', label=str(label))
            ax[2].plot(ticks,aa, str(colours[i % len(colours)]) + str(shapes[(i) % len(shapes)]) + '-', label=str(label))





    for a in ax:
        a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        a.yaxis.set_ticks(np.arange(0.3, 1.1, 0.05))
        a.set_xticklabels(ticks)
        a.set_xticks([0.5*a for  a in range(len(ticks))])
        a.legend(loc='lower left')
        a.grid(axis='both')
    plt.savefig('figs/overall.jpg')
    plt.clf()
    plt.cla()
