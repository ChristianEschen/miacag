
import os
import numpy as npS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
from sklearn import metrics
import scipy
from sklearn.metrics import r2_score
import statsmodels.api as sm
from decimal import Decimal


label_name = 'stenose'
prediction_name = 'stenose_pred'
df_plot = {label_name: [0.3, 0.8, 0.5, 0.2, 0.4, 0.10, 0.20, 0.9, 0.1, 1], prediction_name: [0.3, 0.8, 0.5, 0.2, 0.4, 0.10, 0.20, 0.9, 0.1, 1]}
df_plot = pd.DataFrame(data=df_plot)
g = sns.lmplot(x=label_name, y=prediction_name, data=df_plot)
X2 = sm.add_constant(df_plot[label_name])
est = sm.OLS(df_plot[prediction_name], X2)
est2 = est.fit()
r = est2.rsquared
p = est2.pvalues[label_name]
p = '%.2E' % Decimal(p)

for ax, title in zip(g.axes.flat, [label_name]):
    ax.set_title(title)
    ax.set_ylim(bottom=0.)
    ax.text(0.05, 0.85,
            f'R-squared = {r:.3f}',
            fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.9,
            "p-value = " + p,
            fontsize=9,
            transform=ax.transAxes)
    plt.show()
    
    
plt.title(label_name)
plt.savefig(
    'test_scatter.png', dpi=100,
    bbox_inches='tight')
plt.close()