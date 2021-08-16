from tt_circuit import Info, State, MPS, CircuitFid, Gates, Load

N = 60
D = 200
max_rank = 64

info = Info()

mps = MPS(info)
mps.all_zeros_state(N)
gates = Gates(info)
circuit = CircuitFid(gates)

fid_result = []

circuit.evolution(mps, N, D, fid_result, max_rank)

print(fid_result)
print(len(fid_result))

load = Load('Results.xlsx')
sheet_name = 'Two_qubit_fidelity'
load.write_data(sheet_name, 'B', 1, 200, fid_result)
