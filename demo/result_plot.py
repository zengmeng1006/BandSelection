"===================line Plot========================="
import numpy as np

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


p = '/Users/cengmeng/PycharmProjects/zm/BandSelection/demo/acc.npz'
npz = np.load(p)
accs = np.asanyarray(npz['acc'])
n_groups = 5

SPEC_acc = accs[:, 0]
Lap_score_acc = accs[:, 1]
Spa_acc = accs[:, 2]
NDFS_acc = accs[:, 3]
ISSC_acc = accs[:, 4]
SNMF_acc = accs[:, 5]
DSC_acc = accs[:, 6]


classes = [5, 10, 15, 20, 25]


plt.plot(classes, SPEC_acc, color='g', label='SPEC')
plt.plot(classes, Lap_score_acc, color='r', label='Lap_score')
plt.plot(classes, Spa_acc, color='b', label='Spa')
plt.plot(classes, NDFS_acc, color='k', label='NDFS')
plt.plot(classes, ISSC_acc, color='y', label='ISSC')
plt.plot(classes, SNMF_acc, color='m', label='SNMF')
plt.plot(classes, DSC_acc, color='c', label='DSC')

plt.xlabel('classes')
plt.ylabel('acc')
plt.legend()

plt.show()



