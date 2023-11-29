import torch

def standard_gaussian_1d():
	return torch.distributions.normal.Normal(
		torch.tensor([0.0]), torch.tensor([1.0])
	)

def standard_gaussian_2d():
	return torch.distributions.multivariate_normal.MultivariateNormal(
		torch.zeros(2), torch.eye(2)
	)