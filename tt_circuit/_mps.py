import numpy as np
import torch


class MPS(object):
    def __init__(self, info):
        self.tt_cores = []
        self.info = info
        self.N = None
        self.r = []
        self.phys_ind = []

    def all_zeros_state(self, n):
        self.tt_cores = []
        self.r = [1]
        self.phys_ind = []
        for i in range(n):
            self.tt_cores.append(torch.reshape(torch.tensor([1, 0], dtype=self.info.data_type, device=self.info.device),
                                               (1, 2, 1)))
            self.r.append(1)
            self.phys_ind.append(2)
        self.N = n

    def one_qubit_gate(self, u, n):
        core = self.tt_cores[n]
        self.tt_cores[n] = torch.transpose(torch.tensordot(u, core, dims=([1], [1])), 0, 1)

    def two_qubit_gate(self, u, n, max_rank=None):
        u = torch.reshape(u, [2, 2, 2, 2])
        phi = torch.tensordot(self.tt_cores[n], self.tt_cores[n + 1], dims=([2], [0]))
        phi = torch.tensordot(u, phi, dims=([2, 3], [1, 2]))
        phi = torch.transpose(phi, 0, 2)
        phi = torch.transpose(phi, 1, 2)
        unfolding = phi.reshape([self.r[n] * self.phys_ind[n], self.r[n + 2] * self.phys_ind[n + 1]])
        compressive_left, compressive_right = MPS.tt_svd(unfolding, max_rank)
        self.r[n + 1] = compressive_left.size()[1]
        self.tt_cores[n] = torch.reshape(compressive_left, [self.r[n], self.phys_ind[n], self.r[n + 1]])
        self.tt_cores[n + 1] = torch.reshape(compressive_right, [self.r[n + 1], self.phys_ind[n + 1], self.r[n + 2]])
        self.normalization(n)

    def normalization(self, n):
        self.tt_cores[n] = self.tt_cores[n] / self.get_norm()

    def return_full_tensor(self):
        full_tensor = self.tt_cores[0]
        for i in range(1, len(self.tt_cores), 1):
            full_tensor = torch.tensordot(full_tensor, self.tt_cores[i], dims=([-1], [0]))
        full_tensor = full_tensor.reshape(self.phys_ind)
        return full_tensor

    def return_full_vector(self):
        full_tensor = self.return_full_tensor()
        return torch.reshape(full_tensor, (-1, ))

    @staticmethod
    def tt_svd(unfolding, max_rank=None):
        u, s, v = torch.linalg.svd(unfolding, full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        if max_rank is not None:
            u = u[:, 0:max_rank]
            s = s[0:max_rank]
            v = v[0:max_rank, :]
        compressive_left = torch.tensordot(u, torch.diag(s), dims=([1], [0]))
        compressive_right = v
        return compressive_left, compressive_right

    def get_norm(self):
        core_prev = torch.tensordot(self.tt_cores[0], torch.conj(self.tt_cores[0]), dims=([1], [1]))
        for i in range(1, len(self.tt_cores), 1):
            core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([1], [0]))
            core_prev = torch.tensordot(core_prev, torch.conj(self.tt_cores[i]), dims=([2], [0]))
            core_prev = torch.einsum('ijklkn', core_prev)
            core_prev = torch.transpose(core_prev, 1, 2)
        norm_square = core_prev[0][0][0][0]
        norm = torch.abs(torch.sqrt(norm_square))
        return norm

    def scalar_product(self, phi):
        """
            Calculating <phi|psi>
        """
        if len(self.tt_cores) != len(phi.tt_cores):
            raise RuntimeError('Different size of tensors')
        else:
            core_prev = torch.tensordot(self.tt_cores[0], torch.conj(phi.tt_cores[0]), dims=([1], [1]))
            for i in range(1, len(self.tt_cores), 1):
                core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([1], [0]))
                core_prev = torch.tensordot(core_prev, torch.conj(phi.tt_cores[i]), dims=([2], [0]))
                core_prev = torch.einsum('ijklkn', core_prev)
                core_prev = torch.transpose(core_prev, 1, 2)
            scalar_product = core_prev[0][0][0][0]
            return scalar_product

    def get_element(self, list_of_index):
        matrix_list = [self.tt_cores[i][:, index, :] for i, index in enumerate(list_of_index)]
        element = matrix_list[0]
        for matrix in matrix_list[1:]:
            element = torch.tensordot(element, matrix, dims=([1], [0]))
        return element[0][0]

    def fidelity(self, phi):
        """
            Calculating |<phi|psi>|^2
        """
        overlap = self.scalar_product(phi)
        fid = overlap * torch.conj(overlap)
        return float(fid)
