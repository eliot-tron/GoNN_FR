import torch

class SlicingLayer(torch.nn.Module):
    def __init__(self, indices, dim=-1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.indices = torch.tensor(indices)
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.index_select(input=input, index=self.indices, dim=self.dim)


def orthonormalization(
    basis: torch.Tensor,
    metric: torch.Tensor,
    ) -> torch.Tensor:
    """Compute the orthonormalization of the given set of vectors according to the given [scalar_products]
    with the Gram-Schmidt algorithm 

    .. warning:: Not very stable.

    Args:
        basis (torch.Tensor): Batch of basis (..., a, j)
        scalar_products (torch.Tensor): Batch of scalar products (..., i, j)

    Returns:
        torch.Tensor: Orthonormal basis with dimensions (..., a, i) 
    """

    basis = basis.movedim(-2, 0)
    result_basis = torch.zeros_like(basis)
    # result_change = torch.diag_embed(torch.ones_like(basis)[...,0])

    for i, a in enumerate(basis):
        # correction = torch.einsum('a...i, ...ij, ...j, a...k -> a...k', result, scalar_products, a, result) / torch.einsum('a...i, ...ij, a...j -> a...', result, scalar_products, result).unsqueeze(-1)
        # correction = correction.nan_to_num(0.).sum(0)
        correction = torch.einsum('a...i, ...ij, ...j, a...k -> ...k', result_basis, metric, a, result_basis)
        q = a - correction
        norm_q = torch.sqrt(torch.einsum('...i, ...ij, ...j -> ...', q, metric, q))
        norm_q[torch.isclose(norm_q, torch.zeros_like(norm_q))] = 1.
        q = q / norm_q.unsqueeze(-1)
        q = q.nan_to_num(0.)
        result_basis[i] = q

    basis = basis.movedim(0, -2)
    result_basis = result_basis.movedim(0, -2)
    # verification_product = torch.einsum('...ai, ...ij, ...bj -> ...ab', result_basis, scalar_products, result_basis)
    # I = torch.diag_embed(torch.ones_like(verification_product)[...,-1])
    # print(f"Scalar product verification: {verification_product}")
    # print(f"Verification: {torch.isclose(I, verification_product).prod(-1).prod(-1)}")
    return result_basis

# I = torch.eye(4)[None, :].repeat(1,1,1)
# B = torch.eye(4).unsqueeze(0)
# # B = torch.rand_like(I)
# B[:, -1, :] = 0.
# B[:,-1,:] = - B.sum(-2).clone()
# print(orthonormalization(B,I))
# print(orthonormalization(I,I))

