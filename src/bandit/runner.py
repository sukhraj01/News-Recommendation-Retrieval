"""Prequential (test-then-train) replay under bandit feedback.

For impression i in chronological order: the policy scores/picks using only its
current state, the environment reveals the reward of that ONE pick, and the
observation is applied to the policy -- immediately (`update_interval_s=None`)
or buffered and applied at the next clock-hour boundary
(`update_interval_s=3600`), which is closer to a production system that
retrains/refreshes on a schedule. Either way an update only ever contains
rewards of impressions strictly earlier than the decision being made.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .policies import Policy
from .stream import Arm, BanditEnvironment, ImpressionStream


@dataclass
class RunResult:
    policy: str
    arm: np.ndarray          # chosen index within each impression
    reward: np.ndarray       # revealed reward of the chosen arm
    update_interval_s: float | None
    simulated: bool


def run_prequential(
    policy: Policy,
    stream: ImpressionStream,
    *,
    update_interval_s: float | None = None,
    simulated: bool = False,
    env_seed: int = 0,
    hook=None,
    delay_s: float | None = None,
) -> RunResult:
    env = BanditEnvironment(stream, simulated=simulated, seed=env_seed)
    n = len(stream)
    arms = np.empty(n, dtype=np.int64)
    rewards = np.empty(n, dtype=np.float64)
    assert delay_s is None or update_interval_s is None, "choose either hourly batching or a feedback delay"
    buffer: list[tuple[Arm, float]] = []
    pending: deque = deque()  # (ready_time, observation) for delayed feedback
    next_boundary = None
    for i in range(n):
        t = float(stream.time[i])
        if update_interval_s is not None:
            if next_boundary is None:
                next_boundary = (t // update_interval_s + 1) * update_interval_s
            if t >= next_boundary:
                for arm_obj, r in buffer:
                    policy.update(arm_obj, r)
                buffer.clear()
                next_boundary = (t // update_interval_s + 1) * update_interval_s
        while pending and pending[0][0] <= t:  # feedback whose delay has elapsed by this decision
            policy.update(*pending.popleft()[1])
        k = policy.select(stream.context(i), stream.candidates(i))
        r = env.pull(i, k)
        arms[i], rewards[i] = k, r
        obs = (stream.arm(i, k), r)
        if delay_s is not None:
            pending.append((t + delay_s, obs))
        elif update_interval_s is None:
            policy.update(*obs)
        else:
            buffer.append(obs)
        if hook is not None:
            hook(i, policy)  # read-only observer (diagnostics); called after the update step
    assert env.n_pulls == n, "exactly one reward must be revealed per impression"
    return RunResult(policy.name, arms, rewards, update_interval_s, simulated)
