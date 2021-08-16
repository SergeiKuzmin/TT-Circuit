import numpy as np
import matplotlib.pyplot as plt
from tt_circuit import Load

N = 15

load = Load('Results.xlsx')
sheet_name = 'Porter_Thomas'
result_px_2 = load.read_data(sheet_name, 'A', 1, 5000)
result_px_8 = load.read_data(sheet_name, 'B', 1, 5000)
result_px_32 = load.read_data(sheet_name, 'C', 1, 5000)

data_x_2 = np.array(sorted(result_px_2))
data_y_2 = np.array(range(0, len(data_x_2), 1)) / len(data_x_2)
data_x_2 = data_x_2 * (2 ** N)

data_x_8 = np.array(sorted(result_px_8))
data_y_8 = np.array(range(0, len(data_x_8), 1)) / len(data_x_8)
data_x_8 = data_x_8 * (2 ** N)

data_x_32 = np.array(sorted(result_px_32))
data_y_32 = np.array(range(0, len(data_x_32), 1)) / len(data_x_32)
data_x_32 = data_x_32 * (2 ** N)

data_x_exact = data_x_2 / (2 ** N)
data_y_exact = 1 - ((1 - data_x_exact) ** (2 ** N - 1))
data_x_exact = data_x_exact * (2 ** N)

fig, ax = plt.subplots()
plt.plot(data_x_2, data_y_2, lw=1, alpha=1, label=r'$\chi = 2$')
plt.plot(data_x_8, data_y_8, lw=1, alpha=1, label=r'$\chi = 8$')
plt.plot(data_x_32, data_y_32, lw=1, alpha=1, label=r'$\chi = 32$')
plt.plot(data_x_exact, data_y_exact, '--', lw=3, alpha=1.0, label='Porter-Thomas')
# plt.tick_params(which='major', direction='in')
# plt.tick_params(which='minor', direction='in')
plt.legend(loc='lower right')
ax.minorticks_off()
plt.xlim(0, 6)
plt.ylim(0, 1.0)
# plt.xscale('log')
plt.xlabel(r'$2^N \rho$', fontsize=15)
plt.ylabel(r'$P(p_x < \rho)$', fontsize=15)
# plt.title('Упражнение 1')
plt.show()
