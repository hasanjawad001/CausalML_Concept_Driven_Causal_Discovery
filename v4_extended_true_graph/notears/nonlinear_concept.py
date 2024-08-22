from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True, mask=None, w_threshold=0.3, learned_model=None, metainfo={}):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        if learned_model is None:
            self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
            self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        else:
            self.fc1_pos = learned_model.fc1_pos
            self.fc1_neg = learned_model.fc1_neg

        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()

        self.mask = mask
        self.w_threshold = w_threshold
        self.metainfo = metainfo
        n_zero, n_ineq = np.sum(self.mask==0), np.sum(self.mask==1)
        self.slack_zero = nn.Parameter(torch.zeros(n_zero))
        self.slack_ineq = nn.Parameter(torch.zeros(n_ineq))
        self.slack_zero.bounds = [(0, None)] * n_zero
        self.slack_ineq.bounds = [(0, None)] * n_ineq

        # fc2: local linear layers
        if learned_model is None:
            layers = []
            for l in range(len(dims) - 2):
                layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
            self.fc2 = nn.ModuleList(layers)
        else:
            self.fc2 = learned_model.fc2

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg


    def flat_to_con(self, A):
        
        concepts = self.metainfo['concepts']
        dflat = sum(concepts)
        dcon = len(concepts)
        
        Arow = torch.zeros((dcon,dflat))
        Ad = torch.zeros((dcon,dcon))

        end_concept = np.cumsum(concepts)
        start_i = 0
        for i in range(dcon):
            end_i = end_concept[i]
            Arow[i,:] = (A[start_i:end_i,:].sum(axis=0))/(end_i-start_i)
            start_i = end_i

        start_i = 0
        for i in range(dcon):
            end_i = end_concept[i]
            Ad[:,i] = (Arow[:,start_i:end_i].sum(axis=1))/(end_i-start_i)
            start_i = end_i
       
        return Ad
    
    
    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        ####################################
        # A = self.flat_to_con(A)
        ####################################        
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
    
    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        ####################################
        A = self.flat_to_con(A)
        dcon = self.metainfo['dcon']
        h = trace_expm(A) - dcon  # (Zheng et al. 2018) ## method 1
        # A = self.flat_to_con(A)
        # dcon = self.metainfo['dcon']
        # M = torch.eye(dcon) + A / dcon  # (Yu et al. 2019) ## method 2
        # E = torch.matrix_power(M, dcon - 1)
        # h = (E.t() * M).sum() - dcon
        ####################################                
        # h = trace_expm(A) - d  # (Zheng et al. 2018) ## method 1
        # M = torch.eye(d) + A / d  # (Yu et al. 2019) ## method 2
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def mask_out(self, W, val):
        mask_tensor = torch.Tensor(self.mask == val).bool()
        h_masked = torch.masked_select(W, mask_tensor)
        return h_masked

    def h_zero_func(self):
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        ####################################
        A = self.flat_to_con(A)
        ####################################                
        h_zero_W = self.mask_out(A, 0)
        h_zero = h_zero_W + self.slack_zero - self.w_threshold * self.w_threshold
        return h_zero

    def h_ineq_func(self):
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        ####################################
        A = self.flat_to_con(A)
        ####################################                
        h_ineq_W = self.mask_out(A, 1)
        h_ineq = -h_ineq_W + self.slack_ineq + self.w_threshold * self.w_threshold
        return h_ineq




class NotearsSobolev(nn.Module):
    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, self.d - 1)
        # h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho_acyclic, rho_zero, rho_ineq, alpha, h, rho_max, alpha_zero, alpha_ineq, h_zero_old, h_ineq_old):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    h_zero_new, h_ineq_new = None, None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    # while rho_acyclic < rho_max or/and rho_zero < rho_max or/and rho_ineq < rho_max:
    while rho_acyclic < rho_max:
        # print("--------------------in while loop (rho_acyclic,rho_zero,rho_ineq,rho_max) : (", rho_acyclic,rho_zero,rho_ineq,rho_max, ")")
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)

            h_val = model.h_func()
            penalty = 0.5 * rho_acyclic * h_val * h_val + alpha * h_val
            h_zero_val = model.h_zero_func()
            penalty_zero = 0.5 * rho_zero * torch.sum(h_zero_val * h_zero_val) + torch.sum(alpha_zero * h_zero_val)
            h_ineq_val = model.h_ineq_func()
            penalty_ineq = 0.5 * rho_ineq * torch.sum(h_ineq_val * h_ineq_val) + torch.sum(alpha_ineq * h_ineq_val)

            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            # print(h_zero_val, h_ineq_val)
            # print(loss, penalty, penalty_zero, penalty_ineq, l2_reg, l1_reg)
            primal_obj = loss + penalty + penalty_zero + penalty_ineq + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        # print(model.fc1_to_adj())
        # print()
        with torch.no_grad():
            h_new = model.h_func().item()

            h_zero = model.h_zero_func().cpu().detach().numpy()
            h_ineq = model.h_ineq_func().cpu().detach().numpy()
            # taking absolute value so that violation measure does not get cancelled out
            h_zero_new = np.sum(np.abs(h_zero))
            h_ineq_new = np.sum(np.abs(h_ineq))
            # print(h_new, h, h_zero_new, h_zero_old, h_ineq_new, h_ineq_old)


        if (h_zero_new > 0.25 * h_zero_old):
            if rho_zero < rho_max:
                rho_zero *= 10
        if (h_ineq_new > 0.25 * h_ineq_old):
            if rho_ineq < rho_max:
                rho_ineq *= 10

        if h_new > 0.25 * h:
            rho_acyclic *= 10
        else:
            # print('end while loop! (h_new, h) ', h_new, h)
            break

    alpha += rho_acyclic * h_new
    alpha_zero += torch.from_numpy(rho_zero * h_zero)
    alpha_ineq += torch.from_numpy(rho_ineq * h_ineq)
    return rho_acyclic, rho_zero, rho_ineq, alpha, h_new, alpha_zero, alpha_ineq, h_zero_new, h_ineq_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      # mask=None,
                      # w_threshold: float = 0.3
                      ):
    rho_acyclic, rho_zero, rho_ineq, alpha, h = 1.0, 1.0, 1.0, 0.0, np.inf

    h_zero_old, h_ineq_old = np.inf, np.inf
    n_zero, n_ineq = np.sum(model.mask==0), np.sum(model.mask==1)
    # alpha_zero = np.zeros(n_zero)
    # alpha_ineq = np.zeros(n_ineq)
    alpha_zero = torch.zeros(n_zero)
    alpha_ineq = torch.zeros(n_ineq)

    # print(' initial state @@@@@@@@@@@@')
    # W_est = model.fc1_to_adj()
    # W_est[np.abs(W_est) < model.w_threshold] = 0
    # print(W_est)
    # print('@@@@@@@@@@@@@@@@@@@@@@')


    for _ in range(max_iter):
        print("-----iteration no: ", _)
        rho_acyclic, rho_zero, rho_ineq, alpha, h, alpha_zero, alpha_ineq, h_zero_old, h_ineq_old = dual_ascent_step(
            model, X, lambda1, lambda2, rho_acyclic, rho_zero, rho_ineq, alpha, h, rho_max, alpha_zero, alpha_ineq, h_zero_old, h_ineq_old
        )
        # if h <= h_tol or rho_acyclic >= rho_max or rho_zero >= rho_max or rho_ineq >= rho_max:
        if h <= h_tol or rho_acyclic >= rho_max:
#             print("-----end iteration with (h,h_tol,rho_acyclic,rho_zero,rho_ineq,rho_max) : (", h,h_tol,rho_acyclic,rho_zero,rho_ineq,rho_max, ")")
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < model.w_threshold] = 0

    res = {}
    res['h'] = h
    res['h_zero'] = h_zero_old
    res['h_ineq'] = h_ineq_old
    res['learned_model'] = model
    return W_est, res


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import notears.utils as ut
    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('W_true.csv', B_true, delimiter=',')

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)


if __name__ == '__main__':
    main()
