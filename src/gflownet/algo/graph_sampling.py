import copy
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    action_type_to_mask,
)
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.utils.misc import get_worker_device, get_worker_rng


def relabel(g: Graph, ga: GraphAction):
    """Relabel the nodes for g to 0-N, and the graph action ga applied to g.
    This is necessary because torch_geometric and EnvironmentContext classes expect nodes to be
    labeled 0-N, whereas GraphBuildingEnv.parent can return parents with e.g. a removed node that
    creates a gap in 0-N, leading to a faulty encoding of the graph.
    """
    rmap = dict(zip(g.nodes, range(len(g.nodes))))
    if not len(g) and ga.action == GraphActionType.AddNode:
        rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
    g = g.relabel_nodes(rmap)
    if ga.source is not None:
        ga.source = rmap[ga.source]
    if ga.target is not None:
        ga.target = rmap[ga.target]
    return g, ga


class GraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""

    def __init__(
        self, ctx, env, max_len, max_nodes, sample_temp=1, correct_idempotent=False, pad_with_terminal_state=False
    ):
        """
        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Optional[Tensor], random_action_prob: float = 0.0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
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
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - bck_a: List[GraphAction], the reverse actions
           - is_sink: List[int], sink states have P_B = 1
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
           - interm_rewards: List[float], intermediate rewards
        """
        dev = get_worker_device()
        rng = get_worker_rng()
        # This will be returned
        data = [
            {
                "traj": [],
                "bck_a": [],  # The reverse actions
                "is_valid": True,
                "is_sink": [],
                "fwd_logprobs": [],
                "bck_logprobs": [],
                "interm_rewards": [],
            }
            for _ in range(n)
        ]
        graphs = [self.env.new() for _ in range(n)]
        done = [False for _ in range(n)]

        for t in range(self.max_len):
            self._forward_step(model, data, graphs, cond_info, t, done, rng, dev, random_action_prob)
            if all(done):
                break

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        #
        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]
        #
        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1
        
        self._wrap_up_fwd_trajs(data, graphs)
        return data

    def _wrap_up_fwd_trajs(self, data, graphs):
        for i in range(len(data)):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]["fwd_logprobs"] = torch.stack(data[i]["fwd_logprobs"]).reshape(-1)
            data[i]["bck_logprobs"] = torch.stack(data[i]["bck_logprobs"]).reshape(-1)
            data[i]["fwd_logprob"] = data[i]["fwd_logprobs"].sum()
            data[i]["bck_logprob"] = data[i]["bck_logprobs"].sum()
            data[i]["result"] = graphs[i]
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Pad)))
                data[i]["is_sink"].append(1)

    def sample_backward_from_graphs(
        self,
        graphs: List[Graph],
        model: Optional[nn.Module],
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
    ):
        """Sample a model's P_B starting from a list of graphs, or if the model is None, use a uniform distribution
        over legal actions.

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated
        random_action_prob: float
            Probability of taking a random action (only used if model parameterizes P_B)

        """
        starting_graphs = list(graphs)
        dev = get_worker_device()
        n = len(graphs)
        done = [False] * n
        data = [
            {
                "traj": [(graphs[i], GraphAction(GraphActionType.Stop))],
                "is_valid": True,
                "is_sink": [1],
                "bck_a": [GraphAction(GraphActionType.Pad)],
                "bck_logprobs": [0.0],
                "result": graphs[i],
            }
            for i in range(n)
        ]

        # TODO: This should be doable.
        if random_action_prob > 0:
            warnings.warn("Random action not implemented for backward sampling")

        while sum(done) < n:
            self._backward_step(model, data, graphs, cond_info, done, dev)
        for i in range(n):
            # See comments in sample_from_model
            data[i]["traj"] = data[i]["traj"][::-1]
            data[i]["bck_a"] = [GraphAction(GraphActionType.Pad)] + data[i]["bck_a"][::-1]
            data[i]["is_sink"] = data[i]["is_sink"][::-1]
            data[i]["bck_logprobs"] = torch.tensor(data[i]["bck_logprobs"][::-1], device=dev).reshape(-1)
            if self.pad_with_terminal_state:
                data[i]["traj"].append((starting_graphs[i], GraphAction(GraphActionType.Pad)))
                data[i]["is_sink"].append(1)
        return data

    def local_search_sample_from_model(
        self,
        model: nn.Module,
        n: int,
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
        num_ls_steps: int = 1,
        num_bck_steps: int = 1,
    ):
        dev = get_worker_device()
        rng = get_worker_rng()
        # First get n trajectories
        current_trajs = self.sample_from_model(model, n, cond_info, random_action_prob)
        # Then we're going to perform num_ls_steps of local search, each with num_bck_steps backward steps.
        # Each local search step is a kind of Metropolis-Hastings step, where we propose a new trajectory, which may
        # be accepted or rejected based on the forward and backward probabilities and reward.
        # Finally, we return all the trajectories that were sampled.
        returned_trajs = current_trajs

        for mcmc_steps in range(num_ls_steps):
            # Create new trajectories from the initial ones
            bck_trajs = self.sample_backward_from_graphs(
                [i["result"] for i in current_trajs], model, cond_info, random_action_prob
            )
            # Now we truncate the trajectories, we want to remove at most the last num_bck_steps steps, and also
            # remove the last step(s) if it is a stop (and/or pad) action.
            stop = GraphActionType.Stop
            num_pad = [
                (1 if i["traj"][-1][0].action == stop else 0) + int(self.pad_with_terminal_state) for i in current_trajs
            ]
            trunc_lens = [max(0, len(i["traj"]) - num_bck_steps - pad) for i, pad in zip(current_trajs, num_pad)]
            new_trajs = [
                {
                    key: list(t[key][:k])
                    for key in ["traj", "bck_a", "is_sink", "fwd_logprobs", "bck_logprobs", "interm_rewards"]
                }
                for t, k in zip(bck_trajs, trunc_lens)
            ]
            # Next we sample new endings for the truncated trajectories
            graphs = [i["traj"][-1][0] for i in new_trajs]
            done = [False] * n

            while not all(done):
                self._forward_step(model, new_trajs, graphs, cond_info, 0, done, rng, dev, random_action_prob)
                done = [d or len(t["traj"]) >= self.max_len for d, t in zip(done, new_trajs)]
            self._wrap_up_fwd_trajs(new_trajs, graphs)
            # We add those new trajectories to the list of returned trajectories
            returned_trajs += new_trajs
            # Finally, we replace the current trajectories with the new ones if they are accepted by MH
            for i in range(n):
                if new_trajs[i]["fwd_logprob"] + new_trajs[i]["bck_logprob"] > current_trajs[i]["fwd_logprob"] + current_trajs[i]["bck_logprob"]:
                    current_trajs[i] = new_trajs[i]
    def _forward_step(self, model, data, graphs, cond_info, t, done, rng, dev, random_action_prob) -> None:
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        n = len(data)
        # Construct graphs for the trajectories that aren't yet done
        torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
        not_done_mask = torch.tensor(done, device=dev).logical_not()
        # Forward pass to get GraphActionCategorical
        # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
        # TODO: compute bck_cat.log_prob(bck_a) when relevant
        batch = self.ctx.collate(torch_graphs)
        batch.cond_info = cond_info[not_done_mask] if cond_info is not None else None
        fwd_cat, *_ = model(batch.to(dev))
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
        if self.sample_temp != 1:
            sample_cat = copy.copy(fwd_cat)
            sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
            actions = sample_cat.sample()
        else:
            actions = fwd_cat.sample()
        graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
        log_probs = fwd_cat.log_prob(actions)
        # Step each trajectory, and accumulate statistics
        for i, j in zip(not_done(range(n)), range(n)):
            data[i]["fwd_logprob"].append(log_probs[j].unsqueeze(0))
            data[i]["traj"].append((graphs[i], graph_actions[j]))
            data[i]["bck_a"].append(self.env.reverse(graphs[i], graph_actions[j]))
            # Check if we're done
            if graph_actions[j].action is GraphActionType.Stop:
                done[i] = True
                data[i]["bck_logprob"].append(torch.tensor([1.0], device=dev).log())
                data[i]["is_sink"].append(1)
            else:  # If not done, try to step the self.environment
                gp = graphs[i]
                try:
                    # self.env.step can raise AssertionError if the action is illegal
                    gp = self.env.step(graphs[i], graph_actions[j])
                    assert len(gp.nodes) <= self.max_nodes
                except AssertionError:
                    done[i] = True
                    data[i]["is_valid"] = False
                    data[i]["bck_logprob"].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                    continue
                if t == self.max_len - 1:
                    done[i] = True
                # If no error, add to the trajectory
                # P_B = uniform backward
                n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                data[i]["bck_logprob"].append(torch.tensor([1 / n_back], device=dev).log())
                data[i]["is_sink"].append(0)
                graphs[i] = gp
            if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                # check if the graph is sane (e.g. RDKit can  construct a molecule from it) otherwise
                # treat the done action as illegal
                data[i]["is_valid"] = False
        # Nothing is returned, data is modified in place

    def _backward_step(self, model, data, graphs, cond_info, done, dev) -> None:
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(len(graphs)))]
        not_done_mask = torch.tensor(done, device=dev).logical_not()
        if model is not None:
            gbatch = self.ctx.collate(torch_graphs).to(dev)
            gbatch.cond_info = cond_info[not_done_mask] if cond_info is not None else None
            _, bck_cat, *_ = model(gbatch)
        else:
            gbatch = self.ctx.collate(torch_graphs)
            action_types = self.ctx.bck_action_type_order
            action_masks = [action_type_to_mask(t, gbatch, assert_mask_exists=True) for t in action_types]
            bck_cat = GraphActionCategorical(
                gbatch,
                raw_logits=[torch.ones_like(m) for m in action_masks],
                keys=[GraphTransformerGFN.action_type_to_key(t) for t in action_types],
                action_masks=action_masks,
                types=action_types,
            )
        bck_actions = bck_cat.sample()
        graph_bck_actions = [
            self.ctx.ActionIndex_to_GraphAction(g, a, fwd=False) for g, a in zip(torch_graphs, bck_actions)
        ]
        bck_logprobs = bck_cat.log_prob(bck_actions)

        for i, j in zip(not_done(range(len(graphs))), range(len(graphs))):
            if not done[i]:
                g = graphs[i]
                b_a = graph_bck_actions[j]
                gp = self.env.step(g, b_a)
                f_a = self.env.reverse(g, b_a)
                graphs[i], f_a = relabel(gp, f_a)
                data[i]["traj"].append((graphs[i], f_a))
                data[i]["bck_a"].append(b_a)
                data[i]["is_sink"].append(0)
                data[i]["bck_logprobs"].append(bck_logprobs[j].item())
                if len(graphs[i]) == 0:
                    done[i] = True
