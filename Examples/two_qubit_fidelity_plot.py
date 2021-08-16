import numpy as np
import matplotlib.pyplot as plt
from tt_circuit import Load

load = Load('Results.xlsx')
sheet_name = 'Two_qubit_fidelity'
fid_result_40 = load.read_data(sheet_name, 'A', 1, 200)
fid_result_60 = load.read_data(sheet_name, 'B', 1, 200)
data_x = np.array(range(0, 200, 1))


fid_mean_result_40 = np.array([fid_result_40[0]] + [np.exp(np.log(fid_result_40[0:d]).mean()) for d in data_x[1:]])
fid_mean_result_60 = np.array([fid_result_60[0]] + [np.exp(np.log(fid_result_60[0:d]).mean()) for d in data_x[1:]])

fig, ax = plt.subplots()
plt.plot(data_x, fid_result_40, lw=1, alpha=1, color='red', label=r'$N = 40, \chi = 64$')
plt.plot(data_x, fid_mean_result_40, '--', lw=3, alpha=1, color='red', label=r'$\langle f_n \rangle, N = 40, \chi = 64$')
plt.plot(data_x, fid_result_60, lw=1, alpha=1, color='magenta', label=r'$N = 60, \chi = 64$')
plt.plot(data_x, fid_mean_result_60, '--', lw=3, alpha=1, color='magenta',
         label=r'$\langle f_n \rangle, N = 60, \chi = 64$')

plt.legend(loc='upper right')
ax.minorticks_off()
plt.xlim(0, 200)
plt.xlabel(r'$D$', fontsize=15)
plt.ylabel(r'$f_n$', fontsize=15)
plt.show()
