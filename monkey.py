import functools

import numpy as np
import chainer


def count_decorator(f, counter_cls):
    @functools.wraps(f)
    def _inner(*args, **kwargs):
        self = args[0]
        with counter_cls() as c:
            y = f(*args, **kwargs)
        name = getattr(f, 'name', self.label)
        print('"{}","{}"'.format(name, c.float_ops))
        return y
    return _inner

def decorate_link(link, counter_cls):
    for child in link._children:
        if hasattr(child, 'children'):
            decorate_link(child, counter_cls)
        child_link = getattr(link, child)
        setattr(link, child, count_decorator(child_link, counter_cls))

def override_fn(counter_cls):
    chainer.function_node.FunctionNode.apply = (
        count_decorator(chainer.function_node.FunctionNode.apply, counter_cls)
    )


def ignore_decorator(f):
    @functools.wraps(f)
    def _inner(*args, **kwargs):
        with np.errstate(over='ignore'):
            y = f(*args, **kwargs)
        return y
    return _inner

def override_bn():
    chainer.functions.normalization.batch_normalization._x_hat = (
        ignore_decorator(
            chainer.functions.normalization.batch_normalization._x_hat
        )
    )
