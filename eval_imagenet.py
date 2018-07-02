import argparse
import warnings

import numpy as np

import chainer
from chainer.links.model.vision.vgg import VGG16Layers
from chainer.links.model.vision.googlenet import GoogLeNet
from chainer.links.model.vision.resnet import ResNet50Layers
from chainer.links.model.vision.resnet import ResNet101Layers
from chainer.links.model.vision.resnet import ResNet152Layers

import monkey  # monkey patch


def main():
    parser = argparse.ArgumentParser(description='evaluate imagenet')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--count-by', choices=(None, 'layers', 'functions'),
                        default=None)
    parser.add_argument('model',
                        choices=('vgg16', 'googlenet',
                                 'resnet50', 'resnet101', 'resnet152'))
    args = parser.parse_args()

    if args.model == 'vgg16':
        model = VGG16Layers(pretrained_model=None)
    elif args.model == 'googlenet':
        model = GoogLeNet(pretrained_model=None)
    elif args.model == 'resnet50':
        model = ResNet50Layers(pretrained_model=None)
    elif args.model == 'resnet101':
        model = ResNet101Layers(pretrained_model=None)
    elif args.model == 'resnet152':
        model = ResNet152Layers(pretrained_model=None)
    # override batch_normalization, don't override in prodution
    if 'resnet' in args.model:
        monkey.override_bn()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    else:
        if chainer.config.use_ideep != "never":
            model.to_intel64()

    if args.count_by is not None and args.gpu > 0:
        warnings.warn("count_by with GPU is not supported", ValueError)

    if args.count_by == 'layers':
        from perf_counter import Counter
        monkey.decorate_link(model, Counter)
    if args.count_by == 'functions':
        from perf_counter import CounterHook
    else:
        from monkey import CounterHook

    image = np.zeros((1, 3, 224, 224), dtype=np.float32)  # dummy image
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            with CounterHook() as counter:
                model.predict(image, oversample=False)
    for fn, float_ops in counter.call_history:
        print('"{}","{}"'.format(fn, float_ops))


if __name__ == '__main__':
    main()
