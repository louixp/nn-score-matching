import torch
import torch.distributions as D

def standard_gaussian_1d():
	return D.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

def standard_gaussian_2d():
	return D.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))

def symmetric_gaussian_mixture_1d():
	mix = D.Categorical(torch.ones(2,))
	comp = D.Normal(torch.tensor([-5., 5.]), torch.ones(2))
	return torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

def symmetric_gaussian_mixture_2d():
	mix = D.Categorical(torch.ones(2,))
	comp = D.Independent(
		D.Normal(torch.tensor([[-5., 5.], [0., 0.]]), torch.ones(2, 2)), 1)
	return torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

def uniform_2d():
	return torch.distributions.uniform.Uniform(torch.zeros(2), torch.ones(2))

class GaussianMixtureStripe:
	# TODO: make this a D.Distribution
	def __init__(self):
		self.uniform_distribution = D.Uniform(0, 4)
		self.gaussian_mixture = symmetric_gaussian_mixture_1d()
	
	def sample(self, sample_shape):
		uniform_samples = self.uniform_distribution.sample(sample_shape)
		gaussian_mixture_samples = self.gaussian_mixture.sample(sample_shape)
		return torch.stack((uniform_samples, gaussian_mixture_samples)).T
	
class BoxedStandardGaussian:
	# TODO: make this a D.Distribution
	def __init__(self):
		self.standard_gaussian = standard_gaussian_2d()
	
	def sample(self, sample_shape):
		assert len(sample_shape) == 1
		samples = self.standard_gaussian.sample((sample_shape[0] * 3, ))
		out_of_bounds = (samples > 1) | (samples < -1)
		out_of_bounds = out_of_bounds[:, 0] | out_of_bounds[:, 1]
		return samples[~out_of_bounds][:sample_shape[0]]
	
class NoisyUniform2D:
	# TODO: make this a D.Distribution
	def __init__(self, noise_level):
		self.uniform = uniform_2d()
		self.standard_gaussian = standard_gaussian_2d()
		self.noise_level = noise_level
	
	def sample(self, sample_shape):
		samples = self.uniform.sample(sample_shape)
		noise = self.standard_gaussian.sample(sample_shape)
		return samples + self.noise_level * noise
	
class BrownianDiffusionJoint:
	def __init__(self, base_distribution, max_temp):
		self.base_distribution = base_distribution
		self.max_temp = max_temp
	
	def sample(self, sample_shape):
		samples = self.base_distribution.sample(sample_shape)
		temps = torch.rand(sample_shape) * self.max_temp
		noise = torch.randn(sample_shape)
		return torch.stack((samples + temps * noise, temps)).T