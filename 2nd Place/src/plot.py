'''
My plot functions
'''

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


andy_theme = {'axes.grid': True,
              'grid.linestyle': '--',
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 14,
              'legend.title_fontsize': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.labelsize': 16,
              'axes.titlesize': 20,
              'figure.dpi': 100}
 
#print( matplotlib.rcParams )
matplotlib.rcParams.update(andy_theme)

#Calibration Chart
def cal_data(prob, true, data, bins, plot=False, figsize=(6,4), save_plot=False):
    cal_dat = data[[prob,true]].copy()
    cal_dat['Count'] = 1
    cal_dat['Bin'] = pd.qcut(cal_dat[prob], bins, range(bins) ).astype(int) + 1
    agg_bins = cal_dat.groupby('Bin', as_index=False)['Count',prob,true].sum()
    agg_bins['Predicted'] = agg_bins[prob]/agg_bins['Count']
    agg_bins['Actual'] = agg_bins[true]/agg_bins['Count']
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(agg_bins['Bin'], agg_bins['Predicted'], marker='+', label='Predicted')
        ax.plot(agg_bins['Bin'], agg_bins['Actual'], marker='o', markeredgecolor='w', label='Actual')
        ax.set_ylabel('Outcome')
        ax.legend(loc='upper left')
        ax.set_xlim([0,bins+1])
        plt.xticks(None)
        plt.tick_params(length=0)
        if save_plot:
            plt.savefig(save_plot, dpi=500, bbox_inches='tight')
        plt.show()
    return agg_bins

# Calibration chart by groups (long format, seaborn small multiple)
def cal_data_group(prob, true, group, data, bins, plot=False, wrap_col=3, sns_height=4, save_plot=False):
    cal_dat = data[[prob,true,group]].copy()
    cal_dat['Count'] = 1
    cal_dat['Bin'] = (cal_dat.groupby(group,as_index=False)[prob]
                        ).transform( lambda x: pd.qcut(x, bins, labels=range(bins))
                        ).astype(int) + 1
    agg_bins = cal_dat.groupby([group,'Bin'], as_index=False)['Count',prob,true].sum()
    agg_bins['Predicted'] = agg_bins[prob]/agg_bins['Count']
    agg_bins['Actual'] = agg_bins[true]/agg_bins['Count']
    agg_long = pd.melt(agg_bins, id_vars=['Bin',group], value_vars=['Predicted','Actual'], 
                       var_name='Type', value_name='Probability')
    if plot:
        d = {'marker': ['o','X']}
        ax = sns.FacetGrid(data=agg_long, col=group, hue='Type', hue_kws=d,
                           col_wrap=wrap_col, despine=False, height=sns_height)
        ax.map(plt.plot, 'Bin', 'Outcome', markeredgecolor="w")
        ax.set_titles("{col_name}")
        ax.set_xlabels("")
        ax.set_xticklabels("")
        ax.axes[0].legend(loc='upper left')
        # Setting xlim in FacetGrid not behaving how I want
        for a in ax.axes:
            a.set_xlim([0,bins+1])
            a.tick_params(length=0)
        if save_plot:
            plt.savefig(save_plot, dpi=500, bbox_inches='tight')
        plt.show()
    return agg_bins