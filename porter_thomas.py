from tt_circuit import Info, State, MPS, Circuit, Gates, Load

N = 15
D = 24
max_rank = 32

info = Info()

mps = MPS(info)
mps.all_zeros_state(N)
gates = Gates(info)
circuit = Circuit(gates)

result_px = []

M = 1000
for i in range(M):
    mps.all_zeros_state(N)
    circuit.evolution([mps], N, D, max_rank=max_rank)
    mps_start = MPS(info)
    mps_start.all_zeros_state(N)
    p_x = mps.fidelity(mps_start)
    print('p_x = ', p_x)
    print(mps.r)
    print(i)
    result_px.append(p_x)

print(result_px)

load = Load('Results.xlsx')
sheet_name = 'Porter_Thomas'
load.write_data(sheet_name, 'C', 4001, 4000 + M, result_px)
