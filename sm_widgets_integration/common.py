import functools
from typing import Callable, Any, TypeVar, Dict

K = TypeVar("K")
V = TypeVar("V")


class StoreWrapper:
    def __init__(self, store: dict, key_deser: Callable[[K], Any], val_deser: Callable[[Any], V]):
        self.store = store
        self.key_deser = key_deser
        self.val_deser = val_deser

    @functools.lru_cache
    def __contains__(self, key):
        return self.key_deser(key) in self.store

    @functools.lru_cache
    def __getitem__(self, key):
        val = self.store[self.key_deser(key)]
        return self.val_deser(val)

    def __setitem__(self, key, val):
        assert False

    def __delitem__(self, key):
        assert False

    def __len__(self):
        assert False

    @functools.lru_cache
    def get(self, key, default=None):
        key = self.key_deser(key)
        if key not in self.store:
            return default
        val = self.store[key]
        return self.val_deser(val)