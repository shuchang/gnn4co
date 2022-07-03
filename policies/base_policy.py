import abc
from abc import abstractmethod
import numpy as np


class BasePolicy(object, metaclass=abc.ABCMeta):

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Queries the policy with observation(s) to get selected action(s)"""
        raise NotImplementedError

    @abstractmethod
    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Runs a learning iteration for the policy"""
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str):
        """Saves the training loss"""
        raise NotImplementedError