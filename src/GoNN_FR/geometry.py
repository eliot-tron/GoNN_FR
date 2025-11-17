"""Module implementing tools to examine the geometry of a model."""
from typing import Literal, get_args

import torch
from torch import nn
from torchdiffeq import odeint, odeint_event
from tqdm import tqdm

from GoNN_FR.utils import orthonormalization

_TASKTYPES = Literal["classification", "regression"]
_DIFFTYPES = Literal["functorch", "legacy", "geomstat"]

class GeometricModel(object):
    
    def __init__(self,
                 network: nn.Module,
                 task: _TASKTYPES="classification",
                 diff_method: _DIFFTYPES="functorch",
                 relu_optim: bool=False,
                 verbose: bool=False,
    ) -> None:
        """
            If diff_method=``legacy``, uses torch.autograd.functional.jacobian.
            relu_optim (Boolean): Optimization of the computation if using only ReLU in the model.

            .. warning: ``legacy`` mode is slower and more memory expansive.
        """

        if task not in get_args(_TASKTYPES): 
            raise ValueError(f"Task {task} is not in {get_args(_TASKTYPES)}")
        if diff_method not in get_args(_DIFFTYPES): 
            raise ValueError(f"Differentiation method {diff_method} is not in {get_args(_DIFFTYPES)}")

        super(GeometricModel, self).__init__()
        self.network = network
        self.task = task
        self.diff_method = diff_method
        self.relu_optim = relu_optim
        # self.network.eval()
        self.verbose = verbose
        self.device = next(self.network.parameters()).device
        self.dtype = next(self.network.parameters()).dtype

    def proba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:

        # if len(eval_point.shape) == 3:  # TODO: find a better way
        #     eval_point = eval_point.unsqueeze(0)
        if self.task == "classification":
            p = torch.nn.Softmax(dim=-1)(self.network(eval_point))
        elif self.task == "regression":
            p = self.network(eval_point)
        else:
            raise NotImplementedError()
        if self.verbose: print(f"proba: {p}")
        return p

    def score(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        if self.task == "classification":
            # if len(eval_point.shape) == 3:  # TODO: find a better way
            #     eval_point = eval_point.unsqueeze(0)
            
            return self.network(eval_point)
        else:
            raise NotImplementedError("Score only defined for classification tasks.")

    def grad_proba(
        self,
        eval_point: torch.Tensor,
        out_index: int, 
    ) -> torch.Tensor:

        if self.diff_method == "functorch":
            p_i = lambda x: self.network(x)[...,out_index]
            grad_proba = torch.vmap(torch.func.grad(p_i))(eval_point)
        elif self.diff_method == "legacy":
            j = self.jac_proba(eval_point)
            grad_proba = j[out_index, :]
        else:
            raise NotImplementedError()

        return grad_proba

    def jac_proba(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False,
    ) -> torch.Tensor:
        """Function computing the matrix :math:`∂_l p_a`.

        Args:
            eval_point (torch.Tensor):
                Batch of points of the input space at which the expression is evaluated.
            create_graph (bool, optional):
                If ``True``, the Jacobian will be computed in a differentiable manner.
                Only active when `legacy` is ``True``.

        Returns:
            torch.Tensor: tensor :math:`∂_l p_a` with dimensions (bs, a, l)
        """

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")

        if self.diff_method == "functorch":
            j = torch.vmap(torch.func.jacrev(self.proba))(eval_point)
            if self.verbose: print(f"shape of j before reshape = {j.shape}")
            j = j.squeeze(1)
            if self.verbose: print(f"shape of j after reshape = {j.shape}")
        elif self.diff_method == "legacy":
            j = torch.autograd.functional.jacobian(self.proba, eval_point, create_graph=create_graph) # TODO: what happens if not batched?
            if self.verbose: print(f"shape of j before reshape = {j.shape}")
            j = j.sum(2)
            # We sum on 2 because it is the batch dimension for dx when the output of the
            # net is (bs, c) and because there is no interactions between batches in the derivative
            # we can sum over this dimension to retrieve the only non zero components.
        else:
            raise NotImplementedError()

        j = j.flatten(2)

        if self.verbose: print(f"shape of j after flatten = {j.shape}")

        return j
    
    def jac_score(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False,
    ) -> torch.Tensor:
        """Function computing the matrix :math:`∂_l s_a`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner. Only active when `legacy` is ``True``.

        Returns:
            torch.Tensor: tensor :math:`∂_l s_a` with dimensions (bs, a, l)
        """
        if self.task != "classification":
            raise NotImplementedError("jac_score only implemented for classification tasks.")

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")

        if self.diff_method == "functorch":
            j = torch.vmap(torch.func.jacrev(self.score))(eval_point)
            if self.verbose: print(f"shape of j before reshape = {j.shape}")
            j = j.squeeze(1)
            if self.verbose: print(f"shape of j after reshape = {j.shape}")
        elif self.diff_method == "legacy":
            j = torch.autograd.functional.jacobian(self.score, eval_point, create_graph=create_graph) # TODO: what happens if not batched?
            if self.verbose: print(f"shape of j before reshape = {j.shape}")
            j = j.sum(2)
            # We sum on 2 because it is the batch dimension for dx when the output of the
            # net is (bs, c) and because there is no interactions between batches in the derivative
            # we can sum over this dimension to retrieve the only non zero components.
        else:
            raise NotImplementedError()

        j = j.flatten(2)

        if self.verbose: print(f"shape of j after flatten = {j.shape}")
        
        return j

    def DIM(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False,
        regularisation: bool=False,
    ) -> torch.Tensor:
        """Function computing the Data Information Metric for the given network. 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner. Only used if self.diff_method = "legacy"
            regularisation (bool, optional): If ``True``, :math:`ε` is added to the kernel directions
                of G in order to have a full rank metric.

        Returns:
            torch.Tensor: tensor :math:`g_ij` with dimensions (bs, i, j).
        """
        
        if self.task == "classification":
            J_s = self.jac_score(eval_point, create_graph=create_graph)
            p = self.proba(eval_point)
            P = torch.diag_embed(p, dim1=1)
            pp = torch.einsum("...i, ...j -> ...ij", p, p)
            
            G = torch.einsum("...ji, ...jk, ...kl -> ...il", J_s, (P - pp), J_s)
        # elif self.task == "regression":  # TODO: find a way to implement output_FIM
        #     J_p = self.jac_proba(eval_point)
        #     F = self.output_FIM(eval_point)
        #     G = torch.einsum("zji, zjk, zkl -> zil", J_p, F, J_p)
        else:
            raise NotImplementedError()
        
        if regularisation:
            C = p.shape[-1]
            eigenvalues, eigenvectors = torch.linalg.eigh(G)
            eps = eigenvalues[..., - (C - 1)] / 2  # eps fixed at the lowest non zero eigenvalue / 2

            epsI = torch.einsum("z, ij -> zij", eps, torch.eye(G.shape[-1]))
            epsI[..., - (C - 1):] = 0
            epsKernel = torch.einsum("zij, zjk, zlk -> zil", eigenvectors, epsI, eigenvectors)
            
            return G + epsKernel

        else:
            return G

    def hessian_gradproba(
        self, 
        eval_point: torch.Tensor,
        method: str='func', # 'relu_optim', 'double_jac', 'torch_hessian'
    ) -> torch.Tensor:
        """Function computing :math:`H(p_a)∂p_b`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.
            method (str): Method to compute the hessian:
                - double_jac: uses double jacobian (slow). (Only in self.diff_method=legacy).
                - torch_hessian: uses torch.autograd.functional.hessian (less slow). (Only in self.diff_method=legacy).

        Returns:
            torch.Tensor: Tensor :math:`(H(p_a)∂p_b)_l` with dimensions (bs, a,b,l).
        """

        if self.relu_optim:
            J_p = self.jac_proba(eval_point)
            J_s = self.jac_score(eval_point)
            P = self.proba(eval_point)
            C = P.shape[-1]
            I = torch.eye(C).unsqueeze(0)
            N = P.unsqueeze(-2).expand(-1, C, -1)
            
            """Compute """
            first_term = torch.einsum("...bi, ...ki, ...ak, ...al -> ...abl", J_p, J_s, (I-N), J_p) 
            
            """Compute """
            second_term = torch.einsum("...a, ...bi, ...ki, ...kl -> ...abl", P, J_p, J_s, J_p )
            
            return first_term - second_term

        if self.diff_method == 'functorch':
            # J_p = torch.vmap(torch.func.jacrev(self.proba))(eval_point) # Doesn't work because of softmax
            J_p = self.jac_proba(eval_point, create_graph=False)
            H_p = torch.vmap(torch.func.hessian(self.proba))(eval_point)
            # H_p = torch.vmap(torch.func.jacrev(self.jac_proba, argnums=0))(eval_point) # Doesn't work because of dim and conv2D
            if len(eval_point.shape) > 2:
                k = len(eval_point.shape) - 1
                H_p = H_p.flatten(-k).flatten(-(k+1), -2)
            H_p = H_p.squeeze(1)
            if self.verbose:
                print(f"H_p dimensions after squeeze: {H_p.shape}")
            h_grad_p = torch.einsum("...alk, ...bk -> ...abl", H_p, J_p)
            return  h_grad_p

        elif self.diff_method == "legacy":
            if method == 'double_jac':
                J_p = self.jac_proba(eval_point)
                J = lambda x: self.jac_proba(x, create_graph=True)
                H_p = torch.autograd.functional.jacobian(J, eval_point).sum(3).flatten(3)  # 3 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
                h_grad_p = torch.einsum("...alk, ...bk -> ...abl", H_p, J_p)
                return  h_grad_p
            elif method == 'torch_hessian':
                J_p = self.jac_proba(eval_point)
                shape = self.proba(eval_point).shape
                H_p = []
                for bs, point in enumerate(tqdm(eval_point)):
                    H_list = []
                    for class_index in range(shape[1]):
                        h_p_i = torch.autograd.functional.hessian(lambda x: self.proba(x)[0, class_index], point)
                        h_p_i = h_p_i.flatten(len(point.shape))
                        h_p_i = h_p_i.flatten(end_dim=-2)
                        H_list.append(h_p_i)
                    H_p.append(torch.stack(H_list))
                H_p = torch.stack(H_p)
                # H_list = torch.stack([torch.stack([torch.autograd.functional.hessian(lambda x: self.proba(x)[bs, i], eval_point[bs]) for i in range(shape[1])]) for bs in range(shape[0])])
                h_grad_p = torch.einsum("...alk, ...bk -> ...abl", H_p, J_p)
                return  h_grad_p
            else:
                raise ValueError(f"Unknown method {method}.")
        else:
            raise NotImplementedError()
             
    def lie_bracket(
        self,
        eval_point: torch.Tensor,
        # approximation: bool=False,
    ) -> torch.Tensor:
        """Function computing :math:`[∂p_a, ∂p_b] = H(p_b)∂p_a - H(p_a)∂p_b`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`[∂p_a, ∂p_b]_l` with dimensions (bs, a,b,l)
        """

        if self.task != "classification":
            raise NotImplementedError()
        # if approximation:
        #     J_x = self.jac_proba(eval_point)
        #     new_point = eval_point.unsqueeze() + J_x
        #     raise NotImplementedError
        
        H_grad = self.hessian_gradproba(eval_point)
        
        return H_grad.transpose(-2, -3) - H_grad

    def jac_dot_product(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing :math:`∂_i(∇p_a^t ∇p_b)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`∂_i(∇p_a^t ∇p_b)` with dimensions (bs, a, b, i).
        """

        H_grad = self.hessian_gradproba(eval_point)

        return H_grad.transpose(-2, -3) + H_grad

    def jac_metric(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing :math:`∂_k G_{i,j}`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ∂_k G_{i,j} with dimensions (bs, i, j, k).
        """

        if self.task != "classification":
            raise NotImplementedError()

        if self.relu_optim:
            J_s = self.jac_score(eval_point)
            J_p = self.jac_proba(eval_point)
            p = self.proba(eval_point)
            pdp = torch.einsum("...a, ...bk -> ...kab", p, J_p)  # p_a ∂_k p_b
            return torch.einsum(
                        "...ai, ...kab, ...bj -> ...ijk",
                        J_s, torch.diag_embed(J_p.mT) - pdp - pdp.mT, J_s
                    )

        if self.diff_method == "functorch":
            jac_metric = torch.vmap(torch.func.jacrev(self.DIM))(eval_point)
            if self.verbose: print(f"shape of j before reshape = {jac_metric.shape}")
            jac_metric.squeeze(3) # shouldn't be necessary
            if self.verbose: print(f"shape of j after reshape = {jac_metric.shape}")
            # TODO: test behavior with torch.func.jacrev

        elif self.diff_method == "legacy":
            G = lambda x: self.DIM(x, create_graph=True)
            jac_metric = torch.autograd.functional.jacobian(G, eval_point)
            if self.verbose: print(f"shape of j before reshape = {jac_metric.shape}")
            jac_metric = jac_metric.sum(3).flatten(3)  # Before reshape: (bs, i, j, bs_, k)  ∂_k G_{i,j}
            if self.verbose: print(f"shape of j after reshape = {jac_metric.shape}")
            # self.verbose=False

        else:
            raise NotImplementedError()
            
        return jac_metric

    def christoffel(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing the Christoffel symbols :math:`Γ_{i,j}^k`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`Γ_{i,j}^k` with dimensions (bs, i, j, k).
        """
        dG = self.jac_metric(eval_point)
        G = self.DIM(eval_point)
        B = dG.permute(0, 3, 1, 2) + dG.permute(0, 1, 3, 2) - dG.permute(0, 3, 2, 1)
        # G_inv = torch.linalg.pinv(G.to(torch.double), hermitian=True).to(self.dtype) # input need to be in double
        # TODO what to do when G becomes zero, or when G_inv diverges?
        try:
            # G shape: (bs, l, k) | B shape: (bs, i, l, j)
            result_lstsq = torch.linalg.lstsq(G.unsqueeze(-3).expand((*G.shape[:-2], B.shape[-3], *G.shape[-2:])), B, rcond=1e-7)
            result_lstsq = result_lstsq.solution / 2
            result_lstsq = result_lstsq.mT # lstsq gives (bs, i, k, j) and we want (bs, i, j, k)
        except:
            print("Warning: lstsq in christoffel raised an error.")
            result_lstsq = torch.zeros_like(dG)
        # B_expanded = B.unsqueeze(-1).expand((*B.shape, G.shape[-2])).mT
        # result_lstsq = torch.linalg.lstsq(G[...,None, None, :, :].expand(B_expanded.shape), B_expanded).solution / 2
        # result_pinv = torch.einsum("...kl, ...ilj -> ...ijk", G_inv, B) / 2
        return result_lstsq
    
    def project_kernel(
        self,
        eval_point: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        J = self.jac_proba(eval_point)
        J_T = J.mT
        # We extract the last component since the sum of the column of J_T is equal to zero
        # -> gives the basis of the kernel of J_T 
        kernel_basis = torch.qr(J_T, some=False).Q[:, J_T.shape[1] - 1:]  
        coefficients = torch.linalg.lstsq(kernel_basis, direction).solution
        displacement = torch.mv(kernel_basis, coefficients)
        return displacement
        
    def project_transverse(
        self,
        eval_point: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        J = self.jac_proba(eval_point)
        J_T = J.mT
        try:
            coefficients = torch.linalg.lstsq(J_T, direction).solution
            displacement = torch.einsum("...la, ...a -> ...l", J_T, coefficients)
        except:
            print("Warning: lstsq in project_transverse raised an error.")
            displacement = direction
        return displacement

    def geodesic(
        self,
        eval_point: torch.Tensor,
        init_velocity: torch.Tensor,
        euclidean_budget: float | None = None,
        full_path: bool=False,
        project_leaf: bool=True,
    ) -> torch.Tensor:
        """Compute the geodesic for the DIM's LC connection with initial velocities [init_velocity] at points [eval_point].

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.
            init_velocity (torch.Tensor): Batch of initial velocities for the geodesic.
            euclidean_budget (float, optional): Euclidean budget for the point. Defaults to None.
            full_path (bool, optional): When True, returns the full geodesic path. Defaults to False.
            project_leaf (bool, optional): When True, projects the velocity to the transverse
                leaf at each step. Defaults to True.

        Returns:
            torch.Tensor: Arrival point of the geodesic with dimensions (bs, i)
        """
        print(f"ɛ={euclidean_budget}")
        if len(init_velocity.shape) > 2:
            init_velocity = init_velocity.flatten(1)
        
        def ode(t, y):
            x, v = y
            christoffel = self.christoffel(x)
            a = -torch.einsum("...i, ...j, ...ijk -> ...k", v, v, christoffel)
            # print(f"|v|={v.norm()}", end='\r')
            if project_leaf:
                v = self.project_transverse(x, v)
                # a = self.project_transverse(x, a)
            # sys.stdout.write("\033[K") 
            # if self.verbose:
            self.iteration += 1
            # print("\033[K", end='\r')
            # print(f"iteration n°{self.iteration}: |v|={v.norm():4e}, |a|={a.norm():4e}", end='\r')
            return (v.reshape(x.shape), a)
        
        if euclidean_budget is None:
            raise NotImplementedError
            y0 = (eval_point, init_velocity) # TODO: wrong dim after bash -> should be flatten ?

            solution_ode = odeint(ode, y0, t=torch.linspace(0., 4., 1000), method="rk4")
            solution_ode_x, solution_ode_v = solution_ode

            return solution_ode_x[-1]

        elif euclidean_budget <= 0.:
            return eval_point

        else:
            self.iteration = 0 
            if self.verbose:
                print(f"eval_point: {eval_point.shape}")
                print(f"init_velocity: {init_velocity.shape}")

            if not full_path:
                print("Geodesic computation starting...")
                solution_ode_x, solution_ode_v = [], []
                for point, vel in tqdm(zip(eval_point, init_velocity)):
                    self.iteration = 0
                    y0 = (point.unsqueeze(0), vel.unsqueeze(0))
                    def euclidean_stop(t, y):
                        x, v = y
                        # print("\033[K", end='\r')
                        # print(f"Iteration n°{self.iteration} - Euclidean norm: {float(euclidean_budget - torch.norm(x - y0[0])):3e}", end='\r')
                        return nn.functional.relu(euclidean_budget - torch.norm(x - y0[0])) * (v.norm() > 1e-7).float()
                    with torch.no_grad():
                        event_t, solution_ode = odeint_event(ode, y0, t0=torch.tensor(0.), event_fn=euclidean_stop, method="euler", options={"step_size": euclidean_budget / 10})
                        # event_t, solution_ode = odeint_event(ode, y0, t0=torch.tensor(0.), event_fn=euclidean_stop, method="adaptive_heun") # too long
                    solution_ode_x.append(solution_ode[0])
                    solution_ode_v.append(solution_ode[1])
                
                solution_ode_x = torch.cat(solution_ode_x, dim=1)
                solution_ode_v = torch.cat(solution_ode_v, dim=1)
                
                return solution_ode_x[-1]
                
            # solution_ivp = solve_ivp(ode, t_span = (0, 2), y0=(eval_point.detach().numpy(), init_velocity.detach().numpy()), method='RK23', events=euclidean_stop if euclidean_budget is not None else None)
            
            # if self.verbose: print(f"event_t: {event_t}")
            raise NotImplementedError
            y0 = (eval_point, init_velocity) # TODO: wrong dim after bash -> should be flatten ?

            solution_ode = odeint(ode, y0, t=torch.linspace(0., int(euclidean_budget * 10), 1000), method="rk4", options={"step_size": euclidean_budget / 100})
            
            # self.verbose = True
            solution_ode_x, solution_ode_v = solution_ode
            if full_path:
                return solution_ode_x.transpose(0, 1)
            
            # return solution_ode_x[-1]
            
            if self.verbose:
                print(f"solution_ode_x: {solution_ode_x.shape}")
                print(f"solution_ode_v: {solution_ode_v.shape}")
                print(f"0 is initial value ? {torch.allclose(solution_ode_x[0], eval_point)} dist: {torch.dist(solution_ode_x[0], eval_point)}")

            # Get last point exceeding the euclidean budget
            admissible_indices = ((solution_ode_x - eval_point.unsqueeze(0)).flatten(2).norm(dim=-1) <= euclidean_budget)
            last_admissible_index = admissible_indices.shape[0] - 1 - admissible_indices.flip(dims=[0]).int().argmax(dim=0)
            last_admissible_solution_x = torch.diagonal(solution_ode_x[last_admissible_index], dim1=0, dim2=1).movedim(-1, 0)
            print(f"Warning: geodesics stoped before reaching ɛ: {(last_admissible_index == admissible_indices.shape[0] -1).float().mean() * 100:.2f}%")
                
            if self.verbose:
                last_admissible_solution_x_loop = torch.zeros_like(eval_point)
                last_admissible_index_loop = torch.zeros(eval_point.shape[0])

                for i, step in enumerate(solution_ode_x):
                    for j, batch in enumerate(step):
                        if (batch - eval_point[j]).norm() <= euclidean_budget:
                            last_admissible_index_loop[j] = i
                            last_admissible_solution_x_loop[j] = batch
                print(f"2 solutions are the same ? {torch.allclose(last_admissible_solution_x, last_admissible_solution_x_loop)}")
                print(f"2 indices of solutions are the same ? {torch.allclose(last_admissible_index.int(), last_admissible_index_loop.int())}")
                        
            return last_admissible_solution_x

    def ang_grad_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing :math:`⟨∇p_a, [∇p_b, ∇p_c]⟩`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`⟨∇p_a, [∇p_b, ∇p_c]⟩` with dimensions (bs, a, b, c)
        """

        G = self.DIM(eval_point)
        J_p = self.jac_proba(eval_point)
        lie = self.lie_bracket(eval_point)

        return torch.einsum("...ai, ...ij, ...bcj -> ...abc", J_p, G, lie) 
    
    def grad_metric(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing :math:`∇p_a(G_x) = J(s)^t A_a J(s)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`∇p_a(G_x)_kl` with dimensions (bs, a, k, l)
        """

        if self.task != "classification":
            raise NotImplementedError()
        if not self.relu_optim:
            raise NotImplementedError()

        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        p = self.proba(eval_point)
        """Compute p_l ∇p_k"""
        p_gradp = torch.einsum("...l, ...ki -> ...ikl", p, J_p)
        
        """Compute δ_kl ∇p_k"""
        delta_gradp = torch.eye(J_p.shape[-2], dtype=self.dtype, device=self.device) * J_p.unsqueeze(-1).transpose(-2, -3)

        return torch.einsum("...ai, ...bk, ...ibc, ...cl -> ...akl", 
                            J_p, J_s, delta_gradp - p_gradp - p_gradp.mT, J_s)
    
    def grad_ang_grad(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing :math:`∇p_a⟨∇p_b, ∇p_c⟩`.

        Args:
            eval_point (torch.Tensor): Batch of points of the 
            input space at which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`∇p_a⟨∇p_b, ∇p_c⟩` with dimensions (bs, a, b, c)
        """

        if self.task != "classification":
            raise NotImplementedError()
        
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("...i, ...j -> ...ij", p, p)
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        grad_G = self.grad_metric(eval_point)

        H_grad = self.hessian_gradproba(eval_point)
        elmt_1 = torch.einsum("...bai, ...di, ...de, ...ej, ...cj -> ...abc", H_grad, J_s, (P - pp), J_s, J_p)
        elmt_2 = torch.einsum("...bk, ...akl, ...cl -> ...abc", J_p, grad_G, J_p)

         
        if self.verbose: 
            print(f"Shape of elmt_1: {elmt_1.shape}")
            print(f"Shape of elmt_2: {elmt_2.shape}")

         
        return elmt_1 + elmt_2 + elmt_1.permute(0, 1, 3, 2) 
    
    def ang_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing :math:`⟨∇_(e_a) e_b, e_c⟩` with :math:`e_a = ∇p_a` using Koszul formula.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`⟨∇_(e_a) e_b, e_c⟩` with dimensions (bs, a, b, c)
        """
        
        elmt_1 = self.grad_ang_grad(eval_point)
        elmt_2 = self.ang_grad_lie(eval_point)
        
        return ( elmt_1 + elmt_1.permute(0, 2, 3, 1) - elmt_1.permute(0, 3, 1, 2) - elmt_2 + elmt_2.permute(0, 2, 3, 1) + elmt_2.permute(0, 3, 1, 2) ) / 2 

    def connection_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the connection form :math:`ω(e_k)` on the basis :math:`e_k = ∇p_k`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`ω^i_j(e_k)` with dimensions (bs, i, j, k) 
        """
        
        C = self.ang_connection(eval_point)
        J_p = self.jac_proba(eval_point)
        G = self.DIM(eval_point)
        G_on_data = torch.einsum("...ai, ...ij, ...bj -> ...ab", J_p, G, J_p)
        # G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G_on_data))
        connection_form = torch.linalg.lstsq(G_on_data.unsqueeze(1), C.mT)
        # print(f"connection form rank = {connection_form.rank}")
        connection_form = connection_form.solution  # shape (bs, k, i, j)
        # print(f"Shape of G_on_data: {G_on_data.shape}\n\t C: {C.shape}\n\t connection_form: {connection_form.shape}")
        # if self.verbose:
        #     print("plotting")
        #     plt.matshow(G_on_data[0].detach().numpy()) 
        #     plt.show()
        #     plt.matshow(G_inv[0].detach().numpy())
        #     plt.show()
        
        # return torch.einsum("zil, zkjl -> zijk", G_inv, C)
        return connection_form.permute(0, 2, 3, 1)  # shape (bs, i, j, k)
    
    def jac_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the jacobian of the connection form.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`∂_l ω^i_j(e_k)` with dimensions (bs, i, j, k, l)
        """

        if self.diff_method == "functorch":
            J_omega = torch.vmap(torch.func.jacrev(self.connection_form))(eval_point)
            # TODO: verify dimensions
        elif self.diff_method == "legacy":
            if self.verbose:
                print(f"GC: shape of eval_point = {eval_point.shape}")
                print(f"GC: shape of output = {self.proba(eval_point).shape}")
            J_omega = torch.autograd.functional.jacobian(self.connection_form, eval_point)
            if self.verbose: print(f"GC: shape of j before reshape = {J_omega.shape}")
            
            J_omega = J_omega.sum(4)  # TODO: vérifier pourquoi on somme sur les batchs de l'entrée
            J_omega = J_omega.reshape(*(J_omega.shape[:4]), -1)
            if self.verbose: print(f"GC: shape of j after reshape = {J_omega.shape}")
        else:
            raise NotImplementedError()
        
        return J_omega

    def connection_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`ω^i_j([e_a, e_b])`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`ω^i_j([e_a, e_b])` with dimensions (bs, i, j, a, b).
        """
        
        if self.task != "classification":
            raise NotImplementedError()
        
        omega = self.connection_form(eval_point)
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        P = self.proba(eval_point)
        C = P.shape[-1]
        I = torch.eye(C).unsqueeze(0)
        N = P.unsqueeze(-2).expand(-1, C, -1)
        # print(N)
        
        elmt_1 = torch.einsum("...bl, ...cl, ...ac, ...ija -> ...ijab", J_p, J_s, (I-N), omega) 

        elmt_2 = torch.einsum("...a, ...bl, ...kl, ...ijk -> ...ijab", P, J_p, J_s, omega)
        
        return (elmt_1 - elmt_2).mT - (elmt_1 - elmt_2)

    def grad_hessian_gradproba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`e_a (H(p_b) ∇p_c)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`e_a (H(p_b) ∇p_c)_l` with dimension (bs, a, b, c, l)
        """

        if self.task != "classification":
            raise NotImplementedError()
        if not self.relu_optim:
            raise NotImplementedError()
        
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        H_grad = self.hessian_gradproba(eval_point)
        P = self.proba(eval_point)
        C = P.shape[-1]
        I = torch.eye(C).unsqueeze(0)
        N = P.unsqueeze(-2).expand(-1, C, -1)
        
        result = - torch.einsum('...ai, ...ki, ...kj, ...bj, ...cl -> ...abcl',
                                 J_p, J_p, J_s, J_p, J_p) \
                + torch.einsum('...ck, ...ki, ...bai, ...cl -> ...abcl',
                               (I-N), J_s, H_grad, J_p) \
                + torch.einsum('...ck, ...ki, ...bi, ...cal -> ...abcl', 
                               (I-N), J_s, J_p, H_grad) \
                - torch.einsum('...ai, ...ci, ...kj, ...bj, ...kl -> ...abcl',
                               J_p, J_p, J_s, J_p, J_p) \
                - torch.einsum('...c, ...ki, ...bai, ...kl -> ...abcl',
                               P, J_s, H_grad, J_p) \
                - torch.einsum('...c, ...ki, ...bi, ...kal -> ...abcl',
                               P, J_s, J_p, H_grad)
        
        return result

    def grad_grad_ang(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`e_a (e_b ⟨e_c, e_d⟩)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`e_a (e_b ⟨e_c, e_d⟩)` with dimensions (bs, a, b, c, d).
        """

        if self.task != "classification":
            raise NotImplementedError()
        if not self.relu_optim:
            raise NotImplementedError()
        
        grad_H_grad = self.grad_hessian_gradproba(eval_point)
        H_grad = self.hessian_gradproba(eval_point)
        G = self.DIM(eval_point)
        J_p = self.jac_proba(eval_point)
        grad_G = self.grad_metric(eval_point)
        
        # elmt_1 := e_a(M_{b,c,d})
        elmt_1 = torch.einsum('...acbl, ...lk, ...dk -> ...abcd', 
                              grad_H_grad, G, J_p) \
                + torch.einsum('...cbk, ...akl, ...dl -> ...abcd',
                               H_grad, grad_G, J_p) \
                + torch.einsum('...cbk, ...kl, ...dal -> ...abcd',
                               H_grad, G, H_grad)  

        """ elmt_2_1 := ∇p_a^T H(p_c)^T e_b(G_x) ∇p_d """
        elmt_2_1 = torch.einsum("...cak, ...bkl, ...dl -> ...abcd",
                                H_grad, grad_G, J_p)

        J_s = self.jac_proba(eval_point)
        p = self.proba(eval_point)
        """Compute p_l ∇p_k"""
        p_gradp = torch.einsum("...l, ...ki -> ...ikl", p, J_p)
        
        """Compute δ_kl ∇p_k"""
        delta_gradp = torch.eye(J_p.shape[-2]) * J_p.unsqueeze(-1).transpose(-2, -3)
        """Compute δ_kl H(p_k) ∇p_a (bs, i, a, l, k)"""
        delta_H_grad = torch.eye(H_grad.shape[-3]) * H_grad.unsqueeze(-1).transpose(-2, -4)
        """Compute ∇p_a^T ∇p_b"""
        gradp_gradp = torch.einsum("...ai, ...bi -> ...ab", J_p, J_p)
        """Compute p_k∇p_b^T ∇p_l"""
        p_gradp_gradp = torch.einsum("...k, ...bl -> ...bkl", p, gradp_gradp).unsqueeze(-4)
        """Compute (∇p_a^T ∇p_l)(∇p_b^T ∇p_k)"""
        four_gradp = torch.einsum("...al, ...bk -> ...abkl", gradp_gradp, gradp_gradp)
        """Compute e_a(A_b)_kl"""
        grad_A = torch.einsum("...bai, ...ikl  -> ...abkl",
                                   H_grad, delta_gradp - p_gradp - p_gradp.mT) \
                    + four_gradp \
                    - 2 * p_gradp_gradp \
                    + torch.einsum("...bi, ...ialk -> ...abkl",
                                   J_p, delta_H_grad) \
                    - torch.einsum("...bi, ...l, ...kai -> ...abkl",
                                   J_p, p, H_grad) \
                    + torch.einsum("...bi, ...ikl -> ...bkl",
                                   J_p, delta_gradp).unsqueeze(-4) \
                    - p_gradp_gradp.mT \
                    - four_gradp.mT 

        """Compute e_a(e_b(G_x))"""
        grad_grad_G = torch.einsum("...ki, ...abkl, ...lj -> ...abij",
                                   J_s, grad_A, J_s)
        """ elmt_2_2 := ∇p_c^T e_a(e_b(G_x)) ∇p_d """
        elmt_2_2 = torch.einsum("...ck, ...abkl, ...dl -> ...abcd",
                                J_p, grad_grad_G, J_p)
        """ elmt_2 := e_a(∇p_c^T e_b(G_x) ∇p_d) """
        elmt_2 =  elmt_2_1 + elmt_2_1.permute(0, 1, 2, 4, 3) + elmt_2_2
        
        return elmt_1 + elmt_1.mT + elmt_2

    def grad_ang_grad_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`e_a (⟨e_b, [e_c, e_d]⟩)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`e_a (⟨e_b, [e_c, e_d]⟩)` with dimensions (bs, a, b, c, d).
        """

        J_p = self.jac_proba(eval_point)
        H_grad = self.hessian_gradproba(eval_point)
        lie = H_grad.transpose(1, 2) - H_grad
        G = self.DIM(eval_point)        
        grad_G = self.grad_metric(eval_point)
        grad_H_gradp = self.grad_hessian_gradproba(eval_point)
        grad_lie = grad_H_gradp.transpose(2, 3) - grad_H_gradp
        
        return torch.einsum("zbal, zlk, zcdk -> zabcd",
                            H_grad, G, lie) \
             + torch.einsum("zbi, zaij, zcdi -> zabcd",
                            J_p, grad_G, lie) \
             + torch.einsum("zbi, zij, zacdj -> zabcd",
                            J_p, G, grad_lie)

    def grad_connection_ang(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`e_a (⟨∇_{e_b} e_c, e_d⟩)`.

        It uses the Koszul formula:
        
        .. math::
            :nowrap:

            \\begin{multline*} 
            2 e_a (⟨∇_{e_b} e_c, e_d⟩) = e_a (e_b ⟨e_c, e_d⟩ + e_c ⟨e_d, e_b⟩ - e_d ⟨e_b, e_c⟩ \\\\
            - ⟨e_b, [e_c, e_d]⟩ + ⟨e_c, [e_d, e_b]⟩ + ⟨e_d, [e_b, e_c]⟩).
            \\end{multline*}

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`e_a (⟨∇_{e_b} e_c, e_d⟩)` with dimensions (bs, a, b, c, d).
        """

        elmt_1 = self.grad_grad_ang(eval_point)
        elmt_2 = self.grad_ang_grad_lie(eval_point)
        
        return (elmt_1
                + elmt_1.permute(0, 1, 3, 4, 2)
                - elmt_1.permute(0, 1, 4, 2, 3)
                - elmt_2 
                + elmt_2.permute(0, 1, 3, 4, 2)
                + elmt_2.permute(0, 1, 4, 2, 3)) / 2

    def grad_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`e_a(ω^i_j(e_b))`.

        It uses the formula:

        .. math::
            e_a (⟨∇_{e_b} e_c, e_d⟩) = \\sum_i e_a (ω_c^i(e_b))⟨e_i,e_d⟩ + \\sum_i ω_c^i(e_b) e_a(⟨e_i,e_d⟩).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`e_a(ω^i_j(e_b))` with dimensions (bs, i, j, a, b).
        """

        grad_connection_ang = self.grad_connection_ang(eval_point)
        grad_ang = self.grad_ang_grad(eval_point)
        connection = self.connection_form(eval_point)
        J_p = self.jac_proba(eval_point)
        G = self.DIM(eval_point)
        G_on_data = torch.einsum("...ai, ...ij, ...bj -> ...ab", J_p, G, J_p)
        # G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G_on_data))

        N = grad_connection_ang - torch.einsum("...icb, ...aid -> ...abcd", connection, grad_ang)
        
        result = torch.linalg.lstsq(G_on_data.unsqueeze(1).unsqueeze(1), N.mT)
        # print(f"grad connection rank = {result.rank}")
        result = result.solution
        
        return result.permute(0, 3, 4, 1, 2)
        # return torch.einsum("zdi, zabci -> zabcd", G_inv, N)
    
    def d_connection_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exterior derivative of the connection form: :math:`dω^i_j(e_a, e_b)`.

        It uses the formula:

        .. math::
            dω^i_j(X,Y) = Xω^i_j(Y) - Yω^i_j(X) - ω^i_j([X,Y]).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`dω^i_j(e_a, e_b)` with dimensions (bs, i, j, a, b).
        """
        
        # J_omega = self.jac_connection(eval_point)
        # print(f"J_omega: {J_omega.shape}")
        # J_p = self.jac_proba(eval_point)
        # elmt_1_old = torch.einsum("zak, zijbk -> zijab", J_p, J_omega)
        elmt_1 = self.grad_connection(eval_point)
        elmt_2 = self.connection_lie(eval_point)
        # mask = ~elmt_1_old.isnan() * ~ elmt_1.isnan()
        # i = 2
        # print(f"Elmt_1_old =\n {elmt_1_old[0,i,i,:4,:4]}")
        # print(f"Elmt_1 =\n {elmt_1[0,i,i,:4,:4]}")
        # print(f"Is it a good estimate for domaga? {'Yes' if torch.allclose(elmt_1[mask], elmt_1_old[mask], equal_nan = True) else 'No'}\n \
        #         Error mean = {(elmt_1_old[mask]-elmt_1[mask]).pow(2).mean()}\n \
        #         Max error = {(elmt_1_old[mask]-elmt_1[mask]).abs().max()} out of {max(elmt_1_old[mask].abs().max(), elmt_1[mask].abs().max())}")
        
        return elmt_1 - elmt_1.mT - elmt_2
        
    def wedge_connection_forms(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exterior product of the connection forms: :math:`\\sum_k ω^i_k(e_a) ∧ ω^k_j(e_b)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`(\\sum_k ω^i_k(e_a) ∧ ω^k_j(e_b))` with dimensions (bs, i, j, a, b).
        """
        
        omega = self.connection_form(eval_point)
        
        elmt = torch.einsum("...ika, ...kjb -> ...ijab", omega, omega)

        return elmt - elmt.mT
    
    def curvature_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the curvature forms :math:`Ω^i_j(e_a, e_b)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`(Ω^i_j(e_a, e_b))` with dimensions (bs, i, j, a, b)
        """

        domega = self.d_connection_form(eval_point)
        wedge = self.wedge_connection_forms(eval_point)
        
        return domega + wedge

    def curvature_form_ONB(
        self,
        eval_point: torch.Tensor,
        ) -> torch.Tensor:
        """Compute the curvature forms :math:`Ω^i_j(E_a, E_b)` with (E_k) an Ortho Normal Basis.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`(Ω^i_j(E_a, E_b))` with dimensions (bs, i, j, a, b)
        """
        J_p = self.jac_proba(eval_point)
        G_p = self.DIM(eval_point)
        E_p = orthonormalization(J_p, G_p)
        transition_p = torch.linalg.lstsq(J_p.mT, E_p.mT).solution.mT
        print(f"Transition matrix is ok: {torch.allclose(torch.einsum('...ik, ...kj -> ...ij', transition_p, J_p), E_p)}")
        Omega = self.curvature_form(eval_point)

        return torch.einsum('...ijcd, ...ac, ...bd -> ...ijab', Omega, transition_p, transition_p)

    def rici_curvature(
        self,
        eval_point: torch.Tensor,
        ) -> torch.Tensor:

        """Compute the Rici curvature form :math:`Ric(e_a, e_b)`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`(Ric(e_a, e_b))` with dimensions (bs, a, b)
        """

        raise NotImplementedError()
        Omega = self.curvature_form_ONB(eval_point)
        return torch.einsum('...kbka -> ...ab')

    def scalar_curvature(
        self,
        eval_point: torch.Tensor,
        ) -> torch.Tensor:

        """Compute the Scalar curvature form :math:`S`.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
                which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor :math:`(S)` with dimensions (bs)
        """

        Omega = self.curvature_form_ONB(eval_point)
        return torch.einsum('...klkl -> ...', Omega)
