from typing import List, Tuple

import torch
import torch.nn as nn
import tqdm


class Score:
	def __init__(
			self, 
			score_approximator: nn.Module,
			optimizer: torch.optim.Optimizer,
			distribution: torch.distributions.distribution.Distribution):
		self.score_approximator = score_approximator
		self.optimizer = optimizer
		self.distribution = distribution
	
	def learn(self, 
		   *,
		   n_samples_per_iter: int,
		   n_iters: int,
		   debug: bool = False
		) -> Tuple[List[float], List[float]]:
		loss_history, grad_norm_history = [], []
		for _ in tqdm.trange(n_iters):
			x_batch = self.distribution.sample()
			x_batch.requires_grad = True
	
			score = self.score_approximator(x_batch)
			jac = torch.autograd.functional.jacobian(
				lambda x: self.score_approximator(x).sum(dim=0), 
				x_batch, 
				create_graph=True
			).permute(1, 0, 2)
			tr_jac = jac[:, 0, 0] + jac[:, 1, 1]
	
			loss = torch.mean(torch.linalg.norm(score, dim=1) ** 2 + 2 * tr_jac)
			loss_history.append(loss.item())

			self.optimizer.zero_grad()
			loss.backward()
	
			grad_norm = torch.nn.utils.clip_grad_norm_(
				self.score_approximator.parameters(), torch.inf)
			grad_norm_history.append(grad_norm)
			self.optimizer.step()

		return loss_history, grad_norm_history

	def sample_langevin(
			self,
			init_samples: torch.Tensor,
			n_steps: int,
			epsilon: float
		) -> torch.Tensor:
		x = init_samples
		for _ in tqdm.trange(n_steps):
			x += self.score_approximator(x) / 2 * epsilon
			x += torch.sqrt(epsilon) * torch.randn_like(x)
		return x
	
	def score(self) -> torch.Tensor:
		pass

	def log_prob(self) -> torch.Tensor:
		pass