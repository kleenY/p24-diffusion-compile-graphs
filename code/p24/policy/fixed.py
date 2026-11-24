"""The three fixed S2 policies (plan 5.2). B=3 requests are handled at their native shape (no padding)."""
from __future__ import annotations

from .base import CAPTURE, COMPILE, EAGER, Action


class Policy:
    name = "base"
    arm = None  # arm whose runner serves this policy (cost-table arm; real runner via runners.make_runner)

    def decide(self, request: dict, state) -> Action:
        raise NotImplementedError


class B0Eager(Policy):
    """B0: all eager (E0)."""
    name, arm = "B0", "E0"

    def decide(self, request, state):
        return Action(EAGER)


class B4Capture(Policy):
    """B4: capture on first encounter (G1, one graph per shape; no budget, no eviction in S2)."""
    name, arm = "B4", "G1"

    def decide(self, request, state):
        return Action(CAPTURE)


class B1Compile(Policy):
    """B1: compile on first encounter (C1: torch.compile dynamic=False, recompile_limit 64)."""
    name, arm = "B1", "C1"

    def decide(self, request, state):
        return Action(COMPILE)


POLICIES = {"B0": B0Eager, "B4": B4Capture, "B1": B1Compile}


def make_policy(name: str) -> Policy:
    return POLICIES[name]()
