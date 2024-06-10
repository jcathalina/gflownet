from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gflownet.envs.synthesis_building_env import Graph
from gflownet.envs.synthesis_building_env import GraphActionType, GraphAction, ActionIndex
from gflownet.algo.graph_sampling import Sampler
from gflownet.utils.misc import get_worker_device, get_worker_rng


class SynthesisSampler(Sampler):
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(self, ctx, env, max_len, pad_with_terminal_state=False):
        """
        Parameters
        ----------
        env: ReactionTemplateEnv
            A reaction template environment.
        ctx: ReactionTemplateEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        pad_with_terminal_state: bool
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 5
        self.pad_with_terminal_state = pad_with_terminal_state

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float = 0.0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        random_action_prob: float
            Probability of taking a random action at each step

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, Action]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for _ in range(n)]

        graphs = [self.env.empty_graph() for _ in range(n)]
        done = [False] * n
        bck_a = [[GraphAction(GraphActionType.Stop)] for _ in range(n)]  # 0 corresponds to GraphActionType.Stop

        rng = get_worker_rng()

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(i, traj_len=t) for i in not_done(graphs)]
            nx_graphs = [g for g in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            fwd_cat, *_, _ = model(
                self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask]
            )
            if random_action_prob > 0:
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
                ).float()
                # Set the logits to some large value to have a uniform distribution
                fwd_cat.logits = [
                    is_random_action[b][:, None] * torch.ones_like(i) * 100 + i * (1 - is_random_action[b][:, None])
                    for i, b in zip(fwd_cat.logits, fwd_cat.batch)
                ]
            actions = fwd_cat.sample(nx_graphs=nx_graphs, model=model)
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a, fwd=True) for g, a in zip(torch_graphs, actions)]
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                    bck_a[i].append(GraphAction(GraphActionType.Stop))
                else:  # If not done, step the self.environment
                    gp = graphs[i]
                    gp = self.env.step(graphs[i], graph_actions[j])
                    bck_a[i].append(self.env.reverse(gp, graph_actions[j]))
                    if t == self.max_len - 1:
                        done[i] = True
                    n_back = self.env.count_backward_transitions(gp)
                    if n_back > 0:
                        bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    else:
                        bck_logprob[i].append(torch.tensor([0.001], device=dev).log())
                    data[i]["is_sink"].append(0)
                    graphs[i] = self.ctx.obj_to_graph(gp)
                if done[i] and len(data[i]["traj"]) < 2:
                    data[i]["is_valid"] = False
            if all(done):
                break
        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        for i in range(n):
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data

    def sample_backward_from_graphs(
        self,
        graphs: List[Graph],
        model: Optional[nn.Module],
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
    ):
        """Sample a model's P_B starting from a list of graphs.

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        random_action_prob: float
            Probability of taking a random action (only used if model parameterizes P_B)

        """
        raise NotImplementedError()
