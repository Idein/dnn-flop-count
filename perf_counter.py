import os
import struct

import perfmon

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
