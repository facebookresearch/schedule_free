# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from collections.abc import MutableMapping

_MISSING = object()


class _ParameterState(MutableMapping):
    """A live combined view; only schedule-free keys go into wrapper storage."""

    def __init__(self, wrapper_state, base_state, keys):
        self.maps = (wrapper_state, base_state)
        self.wrapper_keys = keys

    def __getitem__(self, key):
        return self.maps[0 if key in self.wrapper_keys else 1][key]

    def __setitem__(self, key, value):
        self.maps[0 if key in self.wrapper_keys else 1][key] = value

    def __delitem__(self, key):
        del self.maps[0 if key in self.wrapper_keys else 1][key]

    def __iter__(self):
        return iter(dict.fromkeys((*self.maps[1], *self.maps[0])))

    def __len__(self):
        return len(dict.fromkeys((*self.maps[1], *self.maps[0])))

    def copy(self):
        return dict(self)


class _WrapperState(MutableMapping):
    """Keep base initialization independent of the wrapper's iterate buffers."""

    def __init__(self, base, keys):
        self.base = base
        self.wrapper_keys = keys
        self.buffers = defaultdict(dict)

    def __getitem__(self, param):
        return _ParameterState(self.buffers[param], self.base.state[param],
                               self.wrapper_keys)

    def __setitem__(self, param, state):
        # Copy first, since state may itself be a live view into these mappings.
        state = dict(state)
        self.buffers[param] = {k: v for k, v in state.items()
                               if k in self.wrapper_keys}
        self.base.state[param] = {k: v for k, v in state.items()
                                  if k not in self.wrapper_keys}

    def __delitem__(self, param):
        if param not in self:
            raise KeyError(param)
        self.buffers.pop(param, None)
        self.base.state.pop(param, None)

    def __contains__(self, param):
        return param in self.buffers or param in self.base.state

    def get(self, param, default=None):
        return self[param] if param in self else default

    def setdefault(self, param, default=None):
        if param not in self:
            self[param] = {} if default is None else default
        return self[param]

    def pop(self, param, default=_MISSING):
        if param not in self:
            if default is _MISSING:
                raise KeyError(param)
            return default
        state = self[param].copy()
        del self[param]
        return state

    def __iter__(self):
        return iter(dict.fromkeys((*self.base.state, *self.buffers)))

    def __len__(self):
        return len(dict.fromkeys((*self.base.state, *self.buffers)))


class _WrapperStateMixin:
    def _init_wrapper_state(self, keys):
        self._wrapper_state = _WrapperState(self.base, keys)

    @property
    def state(self):
        return self._wrapper_state

    def state_dict(self):
        # Retain the existing flattened checkpoint format, while leaving the
        # base optimizer's live dictionaries (and its hooks) untouched.
        base_state = self.base.state_dict()
        state = dict(base_state['state'])
        for group, packed_group in zip(self.param_groups, base_state['param_groups']):
            for param, index in zip(group['params'], packed_group['params']):
                buffers = self._wrapper_state.buffers.get(param)
                if buffers:
                    state[index] = dict(state.get(index, {}), **buffers)
        return dict(base_state, state=state)

    def load_state_dict(self, state_dict):
        # Split legacy combined checkpoints before the base optimizer sees them.
        # A parameter with only wrapper buffers must still have empty base state
        # so lazy initialization and base __setstate__ methods work correctly.
        keys = self._wrapper_state.wrapper_keys
        parameter_ids = {index for group in state_dict['param_groups']
                         for index in group['params']}
        base_state = {}
        wrapper_state = {}
        for index, state in state_dict['state'].items():
            if index not in parameter_ids:
                base_state[index] = state
                continue
            base_state[index] = {k: v for k, v in state.items() if k not in keys}
            wrapper_state[index] = {k: v for k, v in state.items() if k in keys}

        result = self.base.load_state_dict(dict(state_dict, state=base_state))
        buffers = defaultdict(dict)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for param, index in zip(group['params'], saved_group['params']):
                if wrapper_state.get(index):
                    buffers[param] = {
                        key: value.to(device=param.device, dtype=param.dtype)
                        for key, value in wrapper_state[index].items()
                    }
        self._wrapper_state.buffers = buffers
        return result
