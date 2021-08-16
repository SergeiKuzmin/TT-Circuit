import numpy as np
import matplotlib.pyplot as plt
from tt_circuit import Load

N = 20

load = Load('Results.xlsx')
sheet_name = 'Multi_qubit_fidelity'

fid_result_two_10 = load.read_data(sheet_name, 'A', 1, 50)
fid_result_multi_10 = load.read_data(sheet_name, 'B', 1, 50)
fid_result_two_20 = load.read_data(sheet_name, 'C', 1, 50)
fid_result_multi_20 = load.read_data(sheet_name, 'D', 1, 50)
fid_result_two_50 = load.read_data(sheet_name, 'E', 1, 25)
fid_result_multi_50 = load.read_data(sheet_name, 'F', 1, 25)

data_x = np.array(range(0, 50, 1))

fid_mean_result_two_10 = np.array([fid_result_two_10[0]] + [np.exp(np.log(fid_result_two_10[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_two_20 = np.array([fid_result_two_20[0]] + [np.exp(np.log(fid_result_two_20[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_two_50 = np.array([fid_result_two_50[0]] + [np.exp(np.log(fid_result_two_50[0:d]).mean()) for d in
                                                            data_x[1:25]])

fid_multi_two_10 = np.array([fid_mean_result_two_10[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_20 = np.array([fid_mean_result_two_20[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_50 = np.array([fid_mean_result_two_50[d] ** ((N - 1) * d / 2) for d in data_x[0:25]])

fig, ax = plt.subplots()
plt.plot(data_x, fid_result_multi_10, lw=3, alpha=1, color='red', label=r'$N = 20, \chi = 10$')
plt.scatter(data_x, fid_multi_two_10, color='red')
plt.plot(data_x, fid_result_multi_20, lw=3, alpha=1, color='blue', label=r'$N = 20, \chi = 20$')
plt.scatter(data_x, fid_multi_two_20, color='blue')
plt.plot(data_x[0:25], fid_result_multi_50, lw=3, alpha=1, color='green', label=r'$N = 20, \chi = 50$')
plt.scatter(data_x[0:25], fid_multi_two_50, color='green')
plt.legend(loc='lower left')
ax.minorticks_off()
plt.xlim(0, 24)
plt.ylim(10 ** (-5), 1.0)
plt.yscale('log')
plt.xlabel(r'$D$', fontsize=15)
plt.ylabel(r'$F$', fontsize=15)
plt.show()
