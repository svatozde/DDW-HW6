import pickle
import numpy as np
import  matplotlib.pyplot as plt
import  matplotlib


ticks = [0,1,2,3,4,5,6]

super_results = {}
with open('pickles/super_correct_matric.pickle', 'rb') as handle:
    super_results =  pickle.load(handle)
    print(super_results)

    font = {'size': 7}

    matplotlib.rc('font', **font)


    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(ncols=3,nrows=6, figsize=(17, 27))
    #fig.tight_layout()
    row = -1
    for k1 in [(0.0,1.0),(1.0,0.0)]:
        for k2 in [1,50,500]:
            row += 1
            confusion_matrix = super_results[k1][k2]

            conf_arr = np.zeros((10, 10))

            ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

            for x, k in enumerate(ticks):
                for y, i in enumerate(ticks):
                    value = 0
                    if k in confusion_matrix and i in confusion_matrix[k]:
                        value = confusion_matrix[k][i]
                    conf_arr[x][y] = value

            # print(conf_arr)

            precision_arr = np.zeros((10, 10))
            racall_arr = np.zeros((10, 10))
            for i in range(0, 10):
                precision_arr[i][i] = conf_arr[i, i] / sum(conf_arr[i, :])
                racall_arr[i][i] = conf_arr[i, i] / sum(conf_arr[:, i])


            # Using matshow here just because it sets the ticks up nicely. imshow is faster.


            axes[row][0].matshow(conf_arr, cmap='Blues', interpolation='nearest')
            axes[row][0].set_xticks(np.arange(len(ticks)))
            axes[row][0].set_yticks(np.arange(len(ticks)))
            axes[row][0].set_xticklabels(ticks)
            axes[row][0].set_yticklabels(ticks)
            axes[row][0].set_ylabel('Computed rating')
            axes[row][0].set_xlabel('Real rating')
            axes[row][0].set_title('Confusion heatmap ' + str(k1) + ' cnt:' + str(k2))
            for (i, j), z in np.ndenumerate(conf_arr):
                axes[row][0].text(j, i, str(int(z)), ha='center', va='center')

            axes[row][1].matshow(precision_arr, cmap='Blues', interpolation='nearest')
            axes[row][1].set_xticks(np.arange(len(ticks)))
            axes[row][1].set_yticks(np.arange(len(ticks)))
            axes[row][1].set_xticklabels(ticks)
            axes[row][1].set_yticklabels(ticks)
            axes[row][1].set_ylabel('Computed rating')
            axes[row][1].set_xlabel('Real rating')
            axes[row][1].set_title('Precission  ' + str(k1) + ' cnt:' + str(k2))
            for i in range(0, 10):
                axes[row][1].text(i, i, "{0:.2f}".format(precision_arr[i][i]), ha='center', va='center')

            axes[row][2].matshow(precision_arr, cmap='Blues', interpolation='nearest')
            axes[row][2].set_xticks(np.arange(len(ticks)))
            axes[row][2].set_yticks(np.arange(len(ticks)))
            axes[row][2].set_xticklabels(ticks)
            axes[row][2].set_yticklabels(ticks)
            axes[row][2].set_ylabel('Computed rating')
            axes[row][2].set_xlabel('Real rating')
            axes[row][2].set_title('Recall ' + str(k1) + ' cnt:' + str(k2))
            for i in range(0, 10):
                axes[row][2].text(i, i, "{0:.3f}".format(racall_arr[i][i]), ha='center', va='center')




    fig.savefig('fig_revorked/mat_all_new.jpg')
    plt.cla()
    plt.clf()
