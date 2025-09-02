import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


file = 'dev7784_imps_0_sample_00000.csv'
header = 'dev7784_imps_0_sample_header_00000.csv'
header_df = pd.read_csv(header, sep=';')
df=pd.read_csv(file, sep=';')
df = df.T
df.columns = df.iloc[3]
df.drop(['chunk', 'timestamp', 'fieldname', 'size'], inplace=True)

samples = list(header_df.history_name)
samples = ['_'.join(s.split('_')[:-1]) for s in samples]
samples = list(set(samples))


c = df.param1.values.T
r = df.param0.values.T
f = df.frequency.values.T

for s in samples:
    plt.figure(figsize=(5, 3), dpi=300)
    idx = header_df.history_name.str.contains(s)
    # break
    c1 = c[idx]
    f1 = f[idx]
    plt.title(s)
    for i in range(len(c1)):
        plt.plot(f1[i], c1[i] * 1e12)

    # plt.legend()
    plt.xlabel('F, kHz')
    plt.ylabel('C, pF')
    plt.xscale('log')
    # break