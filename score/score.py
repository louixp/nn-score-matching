from copy import deepcopy
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import tqdm


class AbstractScore:
	def __init__(
			self, 
			model: nn.Module,
			optimizer: torch.optim.Optimizer,
			distribution: torch.distributions.distribution.Distribution):
		self.model = model 
		self.optimizer = optimizer
		self.distribution = distribution
	
	def learn(self, 
		   *,
		   n_samples_per_iter: int,
		   n_iters: int,
		   keep_best: bool = False,
		   use_tqdm: bool = True,
		) -> Tuple[List[float], List[float]]:
		best_loss, best_model = float('inf'), None
		loss_history, grad_norm_history = [], []
		iters = tqdm.trange(n_iters) if use_tqdm else range(n_iters)
		for _ in iters:
			x_batch = self.distribution.sample((n_samples_per_iter, ))
			x_batch.requires_grad = True

			loss = self._compute_loss(x_batch)
			loss_history.append(loss.item())
			if keep_best and loss < best_loss:
				best_model = deepcopy(self.model)

			self.optimizer.zero_grad()
			loss.backward()
	
			grad_norm = torch.nn.utils.clip_grad_norm_(
				self.model.parameters(), torch.inf)
			grad_norm_history.append(grad_norm)
			self.optimizer.step()

		if keep_best:
			self.model = best_model
		return loss_history, grad_norm_history
	
	def _compute_loss(self, samples: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError
	
	def sample_langevin(
			self,
			init_samples: torch.Tensor,
			n_steps: int,
			epsilon: float,
			use_tqdm: bool = True
		) -> torch.Tensor:
		x = init_samples.clone()
		iters = tqdm.trange(n_steps) if use_tqdm else range(n_steps)
		for _ in iters:
			with torch.no_grad():
				x += self.model(x) / 2 * epsilon
			x += math.sqrt(epsilon) * torch.randn_like(x)
		return x
	
	def score_true(self):
		raise NotImplementedError

	def score_approx(self):
		raise NotImplementedError


class Score2d(AbstractScore):
	def _compute_loss(self, samples: torch.Tensor) -> torch.Tensor:
		score = self.model(samples)
		jac = torch.autograd.functional.jacobian(
			lambda x: self.model(x).sum(dim=0), 
			samples, 
			create_graph=True
		).permute(1, 0, 2)
		tr_jac = jac[:, 0, 0] + jac[:, 1, 1]
		return torch.mean(torch.linalg.norm(score, dim=1) ** 2 + 2 * tr_jac)

	def score_true(
			self, 
			x_min: float, x_max: float, 
			y_min: float, y_max: float, 
			step: float
		) -> Tuple[torch.Tensor, torch.Tensor]:
		return 
	
	def score_approx(
			self, 
			x_min: float, x_max: float, 
			y_min: float, y_max: float, 
			step: float
		) -> Tuple[torch.Tensor, torch.Tensor]:
		grid = torch.meshgrid(
			torch.arange(x_min, x_max, step), torch.arange(y_min, y_max, step))
		grid = torch.stack(grid).reshape(2, -1).T
		with torch.no_grad():
			score = self.model(grid)
		return grid, score
	
class Score1d(AbstractScore):
	def _compute_loss(self, samples: torch.Tensor) -> torch.Tensor:
		if len(samples.shape) == 1:
			samples = torch.unsqueeze(samples, 1)
		score = self.model(samples)
		tr_jac = torch.autograd.functional.jacobian(
			lambda x: self.model(x).sum(), samples, create_graph=True)
		return torch.mean(score ** 2 + 2 * tr_jac)
	
	def score_true(
			self, x_min: float, x_max: float, step: float
		) -> Tuple[torch.Tensor, torch.Tensor]:
		return 
	
	def score_approx(
			self, x_min: float, x_max: float, step: float
		) -> Tuple[torch.Tensor, torch.Tensor]:
		grid = torch.unsqueeze(torch.arange(x_min, x_max, step), 1)
		with torch.no_grad():
			score = self.model(grid)
		return grid, score