import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os


def plot_histogram(x, y, path):
    y = [i[0] for i in y.tolist()[0]]
    #fig = plt.figure()

    plt.barh(x, y)

    plt.suptitle('Attention scores for the respective SOPinstanceUIDs')
    plt.xticks(rotation='82.5')

    plt.savefig(os.path.join(path, 'attentionSOP.png'))
    plt.show()
    plt.close()
    plt.clf()
    
    x = [i for i in range(0, len(x))]
    plt.bar(x, y)

    plt.suptitle('Attention scores for the respective instances')
    plt.xticks(rotation='82.5')

    plt.savefig(os.path.join(path, 'attentionInstance.png'))
    plt.show()
    plt.close()
    plt.clf()
   # plt.savefig(os.path.join(path, 'attention.png'))
    #plt.clf()