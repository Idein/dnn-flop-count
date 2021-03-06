import os
import struct

import perfmon
import chainer

THREAD_ENVS = [
    'OMP_NUM_THREADS',
    'GOTO_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
]


class Counter(object):
    def __init__(self):
        self.events = [
            'FP_ARITH:SCALAR_SINGLE',
            'FP_ARITH:128B_PACKED_SINGLE',
            'FP_ARITH:256B_PACKED_SINGLE',
            'FP_ARITH:512B_PACKED_SINGLE',
        ]
        self.packed = [1, 4, 8, 16]  # to convert 32bit float ops
        self.exit = False
        self.old_env = {}

    def __enter__(self):
        # use PerThreadSession and force single thread
        self.session = perfmon.PerThreadSession(os.getpid(), self.events)
        self.session.start()
        for env in THREAD_ENVS:
            self.old_env[env] = os.environ.get(env, None)
            os.environ[env] = "1"
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.counts = [struct.unpack("L", self.session.read(i))[0]
                       for i, _ in enumerate(self.events)]
        # reset environment variables
        for env in THREAD_ENVS:
            if self.old_env[env] is not None:
                os.environ[env] = self.old_env[env]
            else:
                del os.environ[env]
        # close fds
        for fd in self.session.fds:
            os.close(fd)
        self.exit = True

    @property
    def float_ops(self):
        if not self.exit:
            self.counts = [struct.unpack("L", self.session.read(i))[0]
                           for i, _ in enumerate(self.events)]
        return sum(p * n for p, n in zip(self.packed, self.counts))


class CounterHook(chainer.function_hook.FunctionHook):

    def __init__(self):
        self.events = [
            'FP_ARITH:SCALAR_SINGLE',
            'FP_ARITH:128B_PACKED_SINGLE',
            'FP_ARITH:256B_PACKED_SINGLE',
            'FP_ARITH:512B_PACKED_SINGLE',
        ]
        self.packed = [1, 4, 8, 16]  # to convert 32bit float ops
        self.old_env = {}
        self.call_history = []
        self.total_float_ops = 0

    def _preprocess(self):
        # use PerThreadSession and force single thread
        self.session = perfmon.PerThreadSession(os.getpid(), self.events)
        self.session.start()
        for env in THREAD_ENVS:
            self.old_env[env] = os.environ.get(env, None)
            os.environ[env] = "1"

    def forward_preprocess(self, function, in_data):
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self._preprocess()

    def _postprocess(self, function):
        self.counts = [struct.unpack("L", self.session.read(i))[0]
                       for i, _ in enumerate(self.events)]
        # reset environment variables
        for env in THREAD_ENVS:
            if self.old_env[env] is not None:
                os.environ[env] = self.old_env[env]
            else:
                del os.environ[env]
        # close fds
        for fd in self.session.fds:
            os.close(fd)
        float_ops = sum(p * n for p, n in zip(self.packed, self.counts))
        self.call_history.append((function._impl_name, float_ops))
        self.total_float_ops += float_ops

    def forward_postprocess(self, function, in_data):
        self._postprocess(function)

    def backward_postprocess(self, function, in_data):
        self._postprocess(function)
