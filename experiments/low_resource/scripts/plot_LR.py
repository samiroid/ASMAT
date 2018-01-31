
#%%
import matplotlib as mpl
import sys
sys.path.append("/Users/samir/Dev/projects/ASMAT/")
import ASMAT
from ASMAT.lib import helpers, plots
import matplotlib.pyplot as plt
plt.style.use('bmh')

font = {'family' : 'normal',
        'size'   : 24}
mpl.rc('font', **font)
mpl.rc('legend',fontsize=16)
mpl.rc('xtick',labelsize=22)
mpl.rc('ytick',labelsize=24)
mpl.rc('axes',facecolor="white",labelsize=22)
mpl.rc('text', usetex=False)
DPI=200
PLOT_WIDTH=12.8 * 0.6
PLOT_HEIGHT= 6. * 0.6 #7.2

#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
path = "/Users/samir/Dev/projects/ASMAT/experiments/low_resource/server_results/low_resource.txt"
df = helpers.read_results(path,models=["BOW-BIN","BOE-SUM","BOE-BIN","NaiveBayes","NLSE"],run_ids=["LINEAR","50D"])
f,ax = plt.subplots(1,1,figsize=(PLOT_WIDTH,PLOT_HEIGHT))
# # df = helpers.get_df(my_data)
MIN_Y=0.3
MAX_Y=1.1
plots.plot_df(df,ax,x="dataset",ys=df.columns[1:].tolist(),cols=plots.PALETTE_1,ylabel="avg. F1", min_y=MIN_Y, max_y=MAX_Y,rot=30)
plt.tight_layout()

#####################################################################################
os._exit(1)
#%%

data_decay = [['data', 'NLSE','SVR-$\ell_1$','SVR-RBF','SVR-Linear','SVR-PCA'],
              ['1',  0.85, 0.80, 0.65, 0.43, 0.76], 
              ['0.9',0.85, 0.71, 0.55, 0.48, 0.75],
              ['0.8',0.83, 0.65, 0.53, 0.45, 0.71],
              ['0.7',0.83, 0.63, 0.50, 0.43, 0.7],
              ['0.6',0.80, 0.50, 0.45, 0.43, 0.65],
              ['0.5',0.78, 0.45, 0.43, 0.43, 0.63],
              ['0.4',0.70, 0.40, 0.33, 0.32, 0.62],
              ['0.3',0.64, 0.30, 0.30, 0.30, 0.4],
              ['0.2',0.58, 0.28, 0.20, 0.29, 0.3],
              ['0.1',0.50, 0.2,  0.18, 0.28, 0.28],        
             ]
f_d,ax_d = plt.subplots(1,1,figsize=(PLOT_WIDTH,PLOT_HEIGHT))
df_decay = helpers.get_df(data_decay)
colors_1 = [plots.YELLOW, plots.GRAY_L, plots.GRAY_M, plots.GRAY_S, plots.BLUE_S, plots.BLUE_M, plots.BLUE_M]

plots.plot_decay(df_decay,ax_d,x="data",ys=data_decay[0][1:],cols=colors_1,
           ylabel="Kendal Tau",xlabel="proportion of training data",min_y=0)
plt.tight_layout()
#plt.savefig(PLOTS_PATH+"lex_data_decay.png",format="png",dpi=DPI)




