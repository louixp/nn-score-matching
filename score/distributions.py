import torch
import torch.distributions as D

def standard_gaussian_1d():
	return D.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

def standard_gaussian_2d():
	return D.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))

def symmetric_gaussian_mixture_2d():
	mix = D.Categorical(torch.ones(2,))
	comp = D.Independent(
		D.Normal(torch.tensor([[-5., 5.], [0., 0.]]), torch.ones(2, 2)), 1)
	return torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)