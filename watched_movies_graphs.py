import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.ticker import FormatStrFormatter
#[1,2,3,5,10,50,500]
ticks = [0,1,2,3,4,5,6]
ylables= ['1','2','3','5','10','50','500']
colours =  ['r','g','b','y']
shapes = ['+','o','x']

super_results = {}
with open('pickles/super_correct_results.pickle', 'rb') as handle:
    super_results =  pickle.load(handle)

    fig, ax = plt.subplots(ncols=3, figsize=(10, 5))

    ax[0].set_title('Precision')
    ax[1].set_title('Recall')
    ax[2].set_title('F-Measure')

    for i,k1 in enumerate(super_results):
        label = str(k1)
        pp=[]
        rr=[]
        aa=[]
        tt=[]
        ss = []
        for k2 in super_results[k1]:
            vals =  super_results[k1][k2]
            prec = vals[0]/(vals[0]+vals[2])
            pp.append(prec)
            rec=  vals[0] / (vals[0] + vals[3])
            rr.append(rec)
            aa.append((2*rec*prec)/(rec+prec))
            ss.append(vals[0]+vals[3])



        print(pp)
        print(rr)
        print(aa)

        ax[0].plot(ticks, pp, str(colours[i % len(colours)]) + str(shapes[(i) % len(shapes)]) + '-', label=str(label))
        ax[1].plot(ticks, rr, str(colours[i % len(colours)]) + str(shapes[(i) % len(shapes)]) + '-', label=str(label))
        ax[2].plot(ticks, aa, str(colours[i % len(colours)]) + str(shapes[(i) % len(shapes)]) + '-', label=str(label))


    for a in ax:
        a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        a.yaxis.set_ticks(np.arange(0.0, 0.6, 0.05))
        a.set_xticklabels(ylables)
        a.xaxis.set_ticks(ticks)
        a.set_xlabel('Number of recomanded movies')
        a.legend(loc='upper left')
        a.grid(axis='both')

    ax[0].legend(loc='lower left')
    plt.savefig('fig_revorked/overall_count.jpg')
    plt.clf()
    plt.cla()
