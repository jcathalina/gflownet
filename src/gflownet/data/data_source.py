import numpy as np
import torch
from gflownet.data.replay_buffer import ReplayBuffer
from typing import Any, Callable, Dict, List, NewType, Optional, Protocol, Tuple, Generator
from torch.utils.data import Dataset, IterableDataset

from gflownet.config import Config
from gflownet.utils.misc import get_worker_rng
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
#from gflownet.trainer import GFNAlgorithm, GFNTask


def cycle_call(it):
    while True:
        for i in it():
            yield i


class DataSource(IterableDataset):
    def __init__(
        self,
        cfg: Config,
        ctx: GraphBuildingEnvContext,
        algo, #: GFNAlgorithm,
        task, #: GFNTask,  # TODO: this will cause a circular import
        dev: torch.device,
        replay_buffer: Optional[ReplayBuffer] = None,
        is_algo_eval: bool = False,
        start_at_step: int = 0,
    ):
        """A DataSource mixes multiple iterators into one. These are created with do_* methods."""
        self.iterators: List[Generator] = []
        self.cfg = cfg
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.dev = dev
        self.replay_buffer = replay_buffer
        self.is_algo_eval = is_algo_eval

        self.global_step_count = torch.zeros(1, dtype=torch.int64) + start_at_step
        self.global_step_count.share_memory_()
        self.global_step_count_lock = torch.multiprocessing.Lock()
        self.current_iter = start_at_step
        self.sampling_hooks: List[Callable] = []
        self.active = True

    def add_sampling_hook(self, hook: Callable):
        """Add a hook that is called when sampling new trajectories.

        The hook should take a list of trajectories as input.
        The hook will not be called on trajectories that are sampled from the replay buffer or dataset.
        """
        self.sampling_hooks.append(hook)
        return self

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        self.rng = get_worker_rng()
        its = [i() for i in self.iterators]
        self.algo.set_is_eval(self.is_algo_eval)
        while True:
            with self.global_step_count_lock:
                self.current_iter = self.global_step_count.item()
                self.global_step_count += 1
            iterator_outputs = [next(i, None) for i in its]
            if any(i is None for i in iterator_outputs):
                if not all(i is None for i in iterator_outputs):
                    raise ValueError("Some iterators are done, but not all. You may be mixing incompatible iterators.")
                break
            traj_lists, batch_infos = zip(*iterator_outputs)
            trajs = sum(traj_lists, [])
            # Merge all the dicts into one
            batch_info = {}
            for d in batch_infos:
                batch_info.update(d)
            yield self.create_batch(trajs, batch_info)

    def do_sample_model(self, model, num_samples, keep_samples_in_batch=True):
        if not keep_samples_in_batch:
            assert self.replay_buffer is not None, "Throwing away samples without a replay buffer"

        def iterator():
            while self.active:
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_cond_info(num_samples, t)
                trajs = self.algo.create_training_data_from_own_samples(model, num_samples, cond_info, p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs)  # This is a no-op if there is no replay buffer
                batch_info = self.call_sampling_hooks(trajs)
                yield (trajs, batch_info) if keep_samples_in_batch else ([], {})

        self.iterators.append(iterator)
        return self

    def do_sample_replay(self, num_samples):
        def iterator():
            while self.active:
                trajs = self.replay_buffer.sample(num_samples)
                self.relabel_in_hindsight(trajs)  # This is a no-op if the hindsight ratio is 0
                yield trajs, {}
        show_type(iterator)
        self.iterators.append(iterator)
        return self

    def do_dataset_in_order(self, data, num_samples, backwards_model):
        def iterator():
            for idcs in self.iterate_indices(num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                objs, props = map(list, zip(*[data[i] for i in idcs])) if len(idcs) else ([], [])
                trajs = self.algo.create_training_data_from_graphs(objs, backwards_model, cond_info, p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.set_traj_props(trajs, props)
                self.compute_log_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def do_conditionals_dataset_in_order(self, data, num_samples, model):
        def iterator():
            for idcs in self.iterate_indices(len(data), num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = torch.stack([data[i] for i in idcs])
                trajs = self.algo.create_training_data_from_own_samples(model, num_samples, cond_info, p)
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs)  # This is a no-op if there is no replay buffer
                batch_info = self.call_sampling_hooks(trajs)
                yield trajs, batch_info

        self.iterators.append(iterator)
        return self

    def do_sample_dataset(self, data, num_samples, backwards_model):
        def iterator():
            while self.active:
                idcs = self.sample_idcs(len(data), num_samples)
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                objs, props = map(list, zip(*[data[i] for i in idcs])) if len(idcs) else ([], [])
                trajs = self.algo.create_training_data_from_graphs(objs, backwards_model, cond_info, p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.set_traj_props(trajs, props)
                self.compute_log_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def call_sampling_hooks(self, trajs):
        batch_info = {}
        # TODO: just pass trajs to the hooks and deprecate passing all those arguments
        flat_rewards = torch.stack([t["flat_rewards"] for t in trajs])
        # convert cond_info back to a dict
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs["cond_info"][0]}
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        for hook in self.sampling_hooks:
            batch_info.update(hook(trajs, log_rewards, flat_rewards, cond_info))

    def create_batch(self, trajs, batch_info):
        ci = torch.stack([t["cond_info"]["encoding"] for t in trajs])
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        batch = self.algo.construct_batch(trajs, ci, log_rewards)
        batch.num_online = sum(t["is_online"] for t in trajs)
        batch.num_offline = len(trajs) - batch.num_online
        batch.extra_info = batch_info
        batch.preferences = torch.stack([t["preference"] for t in trajs])
        batch.focus_dir = torch.stack([t["focus_dir"] for t in trajs])

        if self.ctx.has_n():  # Does this go somewhere else? Require a flag? Might not be cheap to compute
            log_ns = [self.ctx.traj_log_n(i["traj"]) for i in trajs]
            batch.log_n = torch.tensor([i[-1] for i in log_ns], dtype=torch.float32)
            batch.log_ns = torch.tensor(sum(log_ns, start=[]), dtype=torch.float32)
        # TODO: find code that depends on batch.flat_rewards and deprecate it
        return batch

    def compute_properties(self, trajs, mark_as_online=False):
        """Sets trajs' flat_rewards and is_valid keys by querying the task."""
        # TODO: refactor flat_rewards into properties
        valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
        # fetch the valid trajectories endpoints
        objs = [self.ctx.graph_to_mol(trajs[i]["result"]) for i in valid_idcs]
        # ask the task to compute their reward
        # TODO: it's really weird that the task is responsible for this and returns a flat_rewards
        # tensor whose first dimension is possibly not the same as the output???
        flat_rewards, m_is_valid = self.task.compute_flat_rewards(objs)
        assert flat_rewards.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
        # The task may decide some of the objs are invalid, we have to again filter those
        valid_idcs = valid_idcs[m_is_valid]
        all_fr = torch.zeros((len(trajs), flat_rewards.shape[1]))
        all_fr[valid_idcs] = flat_rewards
        for i in range(len(trajs)):
            trajs[i]["flat_rewards"] = all_fr[i]
            trajs[i]["is_online"] = mark_as_online
        # Override the is_valid key in case the task made some objs invalid
        for i in valid_idcs:
            trajs[i]["is_valid"] = True

    def compute_log_rewards(self, trajs):
        """Sets trajs' log_reward key by querying the task."""
        flat_rewards = torch.stack([t["flat_rewards"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        log_rewards = self.task.cond_info_to_logreward(cond_info, flat_rewards)
        for i in range(len(trajs)):
            trajs[i]["log_reward"] = log_rewards[i] if trajs[i]["is_valid"] else self.cfg.algo.illegal_action_logreward

    def send_to_replay(self, trajs):
        if self.replay_buffer is not None:
            for t in trajs:
                self.replay_buffer.push(t, t["log_rewards"], t["flat_rewards"], t["cond_info"], t["is_valid"])

    def set_traj_cond_info(self, trajs, cond_info):
        for i in range(len(trajs)):
            trajs[i]["cond_info"] = {k: cond_info[k][i] for k in cond_info}

    def set_traj_props(self, trajs, props):
        for i in range(len(trajs)):
            trajs[i]["flat_rewards"] = props[i]  # TODO: refactor

    def relabel_in_hindsight(self, trajs):
        if self.cfg.replay.hindsight_ratio == 0:
            return
        assert hasattr(
                    self.task, "relabel_condinfo_and_logrewards"
                ), "Hindsight requires the task to implement relabel_condinfo_and_logrewards"
        # samples indexes of trajectories without repeats
        hindsight_idxs = torch.randperm(len(trajs))[: int(len(trajs) * self.cfg.replay.hindsight_ratio)]
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        flat_rewards = torch.stack([t["flat_rewards"] for t in trajs])
        cond_info, log_rewards = self.task.relabel_condinfo_and_logrewards(
            cond_info, log_rewards, flat_rewards, hindsight_idxs
        )
        # TODO: This seems wrong, since we haven't recomputed is_valid
        # log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward

    def sample_idcs(self, n, num_samples):
        return self.rng.choice(n, num_samples, replace=False)

    def iterate_indices(self, n, num_samples):
        worker_info = torch.utils.data.get_worker_info()
        if n == 0:
            # Should we be raising an error here? warning?
            yield np.arange(0, 0)
            return
    
        if worker_info is None:  # no multi-processing
            start, end, wid = 0, n, -1
        else:  # split the data into chunks (per-worker)
            nw = worker_info.num_workers
            wid = worker_info.id
            start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))
        
        if end - start <= num_samples:
            yield np.arange(start, end)
            return
        for i in range(start, end - num_samples, num_samples):
            yield np.arange(i, i + num_samples)
        if i + num_samples < end:
            yield np.arange(i + num_samples, end)