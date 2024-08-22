import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, mask=None, lambda_mask=0.,list_data=[]):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        #     E = slin.expm(W * W)  # (Zheng et al. 2018)
        #     h = np.trace(E) - d
        M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])
    
    ### TODO add knowledge of additional connections
    def _mask_zero(W, mask, val):
        h_zero = W[mask == val]
        no_eq = np.sum(mask==val)

        G_zero = np.zeros((no_eq, W.shape[0], W.shape[1]))
        eq_ind = 0
        for i, row in enumerate(mask):
            for j, x in enumerate(row):
                if x == val:
                    G_zero[eq_ind][i][j] = 1
                    eq_ind = eq_ind + 1
        
        return h_zero, G_zero
    

    def _func(w_all, mask, lambda_mask):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        n_zero, n_ineq = np.sum(mask == 0), np.sum(mask == 1)
        w = w_all[:2*d*d]
        s_zero = w_all[(2*d*d):(2*d*d)+n_zero]
        s_ineq = w_all[(2*d*d)+n_zero:] #s
        
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)

        h_zero_W, G_zero = _mask_zero(W, mask, 0) ## additional loss information
        h_zero = h_zero_W*h_zero_W + s_zero - w_threshold*w_threshold
        h_all_zero = np.sum(h_zero*alpha_zero)
        h_all_zero_sq = np.sum(h_zero*h_zero)
        for i in range(h_zero.shape[0]):
            G_zero[i] = G_zero[i] * h_zero_W[i] * (alpha_zero[i]+rho*h_zero[i])
        G_all_zero = 2*G_zero.sum(axis=0)
        if s_zero.shape[0] == 0:
            G_s_zero = np.zeros(0)
        else:
            G_s_zero = alpha_zero+rho*h_zero

        
        h_ineq_W, G_ineq = _mask_zero(W, mask, 1)
        h_ineq = -h_ineq_W*h_ineq_W + s_ineq + w_threshold*w_threshold
        h_all_ineq = np.sum(h_ineq*alpha_ineq)
        h_all_ineq_sq = np.sum(h_ineq*h_ineq)
        for i in range(h_ineq.shape[0]):
            G_ineq[i] = G_ineq[i] * h_ineq_W[i] * (alpha_ineq[i]+rho*h_ineq[i])
        G_all_ineq = -2*G_ineq.sum(axis=0)
        if s_ineq.shape[0] == 0:
            G_s_ineq = np.zeros(0)
        else:
            G_s_ineq = alpha_ineq+rho*h_ineq
        
        obj = loss + 0.5 * rho * h * h + alpha * h + \
                h_all_zero + 0.5*rho*h_all_zero_sq + \
                h_all_ineq + 0.5*rho*h_all_ineq_sq + \
                lambda1 * w.sum() 
        
        G_smooth = G_loss + (rho * h + alpha) * G_h + G_all_zero + G_all_ineq # dL/dW ; this dW does not include seq, sineq, so its basically dL/d[w+, w-]
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1, G_s_zero, G_s_ineq), axis=None) # [(dL/dwpos= dL/dW * dW/dpos), (dL/dwneg), (dL/dseq), (dL/dineq)]
    
        return obj, g_obj

    n, d = X.shape

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    h_zero_old, h_ineq_old = np.inf, np.inf
    n_zero, n_ineq = np.sum(mask==0), np.sum(mask==1)
    alpha_zero = np.zeros(np.sum(mask==0))
    alpha_ineq = np.zeros(np.sum(mask==1))

    w_zero = np.zeros(np.sum(mask==0))
    bnds_zero = [(0, None)]*np.sum(mask==0)
    w_ineq = np.zeros(np.sum(mask==1))
    bnds_ineq = [(0, None)]*np.sum(mask==1)
    bnds_w = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    bnds = bnds_w + bnds_zero + bnds_ineq
    w_est_all = np.concatenate((w_est, w_zero, w_ineq))
    
    for _ in range(max_iter):
        # print(_)
        w_new, h_new, h_zero_new, h_ineq_new = None, None, None, None
        while rho < rho_max:
            # print(rho)
            sol = sopt.minimize(_func, w_est_all, args=(mask, lambda_mask), method='L-BFGS-B', jac=True, bounds=bnds)


            #w_new = sol.x
            w_new = sol.x[:2*d*d]
            s_zero = sol.x[(2*d*d):((2*d*d)+n_zero)]
            s_ineq = sol.x[((2*d*d)+n_zero):]
            
            # probably this following line needs to change
            h_new, _ = _h(_adj(w_new))
            h_zero_W, _ = _mask_zero(_adj(w_new), mask, 0)
            h_zero = h_zero_W*h_zero_W + s_zero - w_threshold*w_threshold
            h_ineq_W, _ = _mask_zero(_adj(w_new), mask, 1) 
            h_ineq = -h_ineq_W*h_ineq_W + s_ineq + w_threshold*w_threshold

            # elem for list_data
            l1=np.sum(np.absolute(_adj(w_new)))
            elem = (rho,l1,h_new)
            # print(elem)
            list_data.append(elem)

            # taking absolute value so that violation measure does not get cancelled out
            h_zero_new = np.absolute(h_zero).sum()
            h_ineq_new = np.absolute(h_ineq).sum()

#             if (h_new+h_zero_new+h_ineq_new > 0.25 * (h+h_zero_old+h_ineq_old)):
#                 rho *= 10
#             else:
#                 break

            # taking greater_equal to avoid problem with cases where there is no equality or ineq constaints
            # if (h_new > 0.25 * h) and ((n_zero!=0 and h_zero_new>0.25*h_zero_old) or (n_zero==0)) and ((n_ineq!=0 and h_ineq_new>0.25*h_ineq_old) or (n_ineq==0)):
            if (h_new >= 0.25 * h) & \
               (h_zero_new >= 0.25 * h_zero_old) & \
               (h_ineq_new >= 0.25 * h_ineq_old):
                rho *= 5
            else:
                break
                
        w_est_all, h = sol.x, h_new
        h_zero_old, h_ineq_old = h_zero_new, h_ineq_new
        alpha += rho * h
        alpha_zero += rho * h_zero
        alpha_ineq += rho * h_ineq
        
        if h <= h_tol or rho >= rho_max:
            break
    
    W_est = _adj(w_est_all[:2*d*d])
    W_est[np.abs(W_est) < w_threshold] = 0
    
    res = {}
    res['fun'] = sol.fun
    res['h'] = h
    res['h_zero'] = h_zero_new
    res['h_ineq'] = h_ineq_new

    return W_est, res, list_data


if __name__ == '__main__':
    import notears.utils as ut
    ut.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = ut.simulate_dag(d, s0, graph_type)
    W_true = ut.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = ut.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)

