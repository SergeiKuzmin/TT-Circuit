import numpy as np
import torch


class State(object):
    def __init__(self, info):
        self.info = info
        self.N = None
        self.phys_ind = []
        self.full_vector = None

    def all_zeros_state(self, n):
        self.N = n
        self.phys_ind = []
        state_qubit = torch.reshape(torch.tensor([1, 0], dtype=self.info.data_type, device=self.info.device), [1, 2, 1])
        full_tensor = state_qubit
        self.phys_ind.append(2)
        for i in range(1, n, 1):
            full_tensor = torch.tensordot(full_tensor, state_qubit, dims=([-1], [0]))
            self.phys_ind.append(2)
        self.full_vector = torch.reshape(full_tensor, (-1, ))

    def return_full_tensor(self):
        return self.full_vector.reshape(self.phys_ind)

    def return_full_vector(self):
        return self.full_vector

    def one_qubit_gate(self, u, n):
        identity_gate = torch.tensor([[1, 0], [0, 1]], dtype=self.info.data_type, device=self.info.device)
        if n == 0:
            gate = u
        else:
            gate = identity_gate
            for i in range(1, n, 1):
                gate = torch.kron(gate, identity_gate)
            gate = torch.kron(gate, u)
        for i in range(n + 1, self.N, 1):
            gate = torch.kron(gate, identity_gate)
        self.full_vector = torch.mv(gate, self.full_vector)

    def two_qubit_gate(self, u, n, max_rank=None):
        identity_gate = torch.tensor([[1, 0], [0, 1]], dtype=self.info.data_type, device=self.info.device)
        if n == 0:
            gate = u
        else:
            gate = identity_gate
            for i in range(1, n, 1):
                gate = torch.kron(gate, identity_gate)
            gate = torch.kron(gate, u)
        for i in range(n + 2, self.N, 1):
            gate = torch.kron(gate, identity_gate)
        self.full_vector = torch.mv(gate, self.full_vector)

    def get_norm(self):
        return self.full_vector.norm()

    def scalar_product(self, phi):
        """
            Calculating <phi|psi>
        """
        if len(self.full_vector.size()) != len(phi.full_vector.size()):
            raise RuntimeError('Different size of tensors')
        else:
            scalar_product = torch.tensordot(torch.conj(phi.full_vector), self.full_vector, dims=([0], [0]))
            return scalar_product

    def get_element(self, list_of_index):
        return self.full_vector.reshape(self.phys_ind)[tuple(list_of_index)]

    def fidelity(self, phi):
        """
            Calculating |<phi|psi>|^2
        """
        overlap = self.scalar_product(phi)
        fid = overlap * torch.conj(overlap)
        return float(fid)
