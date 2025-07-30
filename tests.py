import torch
from geometry import GeometricModel


def jac_proba_true_xor(geomodel, x):
    W_1 = geomodel.network[0].weight
    b_1 = geomodel.network[0].bias
    W_2 = geomodel.network[2].weight
    p = geomodel.proba(x)
    P = torch.diag_embed(p, dim1=1)
    pp = torch.einsum("...i, ...j -> ...ij", p, p)
    T = torch.heaviside(x @ W_1.T + b_1, torch.zeros_like(b_1))
    return torch.einsum(
        "...ik, ...kh, ...h, ...hj -> ...ij",
        P - pp, W_2, T, W_1
    )

def test_jac_proba(
    geomodel: GeometricModel,
    eval_point: torch.Tensor,
) -> None:
    J_true = jac_proba_true_xor(geomodel, eval_point)
    J = geomodel.jac_proba(eval_point)
    p = geomodel.proba(eval_point)
    P = torch.diag_embed(p, dim1=1)
    pp = torch.einsum("...i, ...j -> ...ij", p, p)
    J_from_score = torch.einsum(
        "...ik, ...kj -> ...ij",
        P - pp, geomodel.jac_score(eval_point)
    )
    good_estimate = torch.isclose(J, J_true).all()
    print(f"Is jac_proba a good estimate for the jacobian?\
            {'Yes' if good_estimate else 'No'}\n \
            Error mean = {(J_true-J).abs().mean()}\n \
            Max error = {(J_true-J).abs().max()} out of {torch.max(J_true.abs().max(), J.abs().max())}")
    
    good_estimate = torch.isclose(J, J_from_score).all()
    print(f"Is jac_from_score a good estimate for the jacobian?\
            {'Yes' if good_estimate else 'No'}\n \
            Error mean = {(J_from_score-J).abs().mean()}\n \
            Max error = {(J_from_score-J).abs().max()} out of {torch.max(J_from_score.abs().max(), J.abs().max())}")
