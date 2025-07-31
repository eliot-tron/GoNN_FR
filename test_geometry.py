import torch
import pytest
from geometry import GeometricModel

network = torch.nn.Sequential(torch.nn.Linear(2,16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
@pytest.fixture
def geomodel():
    return GeometricModel(network, task= "classification", diff_method="functorch")
@pytest.fixture
def eval_point():
    return torch.rand(20, 2)

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

def test_jac_proba_XOR(
    geomodel: GeometricModel,
    eval_point: torch.Tensor,
):
    J_true = jac_proba_true_xor(geomodel, eval_point)
    J = geomodel.jac_proba(eval_point)
    good_estimate = torch.isclose(J, J_true).all()
    assert good_estimate
    # print(f"Is jac_proba a good estimate for the jacobian?\
    #         {'Yes' if good_estimate else 'No'}\n \
    #         Error mean = {(J_true-J).abs().mean()}\n \
    #         Max error = {(J_true-J).abs().max()} out of {torch.max(J_true.abs().max(), J.abs().max())}")
    
def test_jac_from_score(
    geomodel: GeometricModel,
    eval_point: torch.Tensor,
):
    J = geomodel.jac_proba(eval_point)
    p = geomodel.proba(eval_point)
    P = torch.diag_embed(p, dim1=1)
    pp = torch.einsum("...i, ...j -> ...ij", p, p)
    J_from_score = torch.einsum(
        "...ik, ...kj -> ...ij",
        P - pp, geomodel.jac_score(eval_point)
    )
    good_estimate = torch.isclose(J, J_from_score).all()
    assert good_estimate
    # print(f"Is jac_from_score a good estimate for the jacobian?\
    #         {'Yes' if good_estimate else 'No'}\n \
    #         Error mean = {(J_from_score-J).abs().mean()}\n \
    #         Max error = {(J_from_score-J).abs().max()} out of {torch.max(J_from_score.abs().max(), J.abs().max())}")
