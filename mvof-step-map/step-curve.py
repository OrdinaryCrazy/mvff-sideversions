#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import matplotlib.pyplot as plt
# from scipy import interpolate
import numpy as np

# step = np.array([12,     6,      4,      3,      2])
# MAP5 = np.array([0.6480, 0.6797, 0.6898, 0.6921, 0.6982])

# step_new = np.arange(step.min(), step.max(), 0.1)
# # step_new = np.arange(2, 11, 0.1)
# func = interpolate.interp1d(step, MAP5, kind='cubic', fill_value="extrapolate")
# MAP5_new = func(step_new)
# plt.figure(figsize=(10,10))
# ax1 = plt.subplot(2,1,2)
# plt.sca(ax1)

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel("KEY-FRAME STEP", fontsize=16)
# plt.ylabel("MAP@5", fontsize=16)
# plt.title("MVOF STEP-MAP@5 CURVE", fontsize=16)

# plt.plot(step_new, MAP5_new, label="$MVOF\quad MAP@5$", linestyle='--')
# plt.scatter(step, MAP5, color="g")
# plt.hlines(0.7026, 13, 2, colors = "r", linestyles = "--", label="$DFF\qquad MAP@5$")
# plt.legend(loc="lower left", fontsize=16)

# ax2 = plt.subplot(2,1,1)
# plt.sca(ax2)
# the_table = plt.table(cellText=[list(np.flip(step, 0)), list(np.flip(MAP5, 0))],
#                       rowLabels=["STEP", "MAP@5"],
# #                       colLabels=list(np.flip(step, 0)),
#                       loc='lower center')
# the_table.set_fontsize(18)
# the_table.scale(1, 2)
# plt.axis('off')

# plt.show()


# In[4]:


import pickle
diffs = []
mvs = []
flows = []
for i in range(602):
    try:
        flow = pickle.load(open("/home/jingtun/feat_flow_compare/flow_%06d.pkl" % i, 'rb'))
        mv   = pickle.load(open("/home/jingtun/feat_flow_compare/mv_%06d.pkl" % i, 'rb'))
        diff = flow - mv
        diffs.append(np.mean(abs(diff)))
        mvs.append(np.mean(abs(mv)))
        flows.append(np.mean(abs(flow)))
    except:
        print("not fit")
print("diff abs mean : ", np.mean(diffs))
print("mv abs mean : ", np.mean(mvs))
print("flow abs mean : ", np.mean(flows))

