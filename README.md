# dnn-flop-count

Count number of Floating points operations by run, without estimate.
Using performance counter on the processors:

- On Intel CPU (only work on Skylake/Kabylake), using [perfmon2](http://perfmon2.sourceforge.net/) with disabling multi-threads

## On Intel CPU (Skylake)

### Example

This may only works on Skylake/Kabylake and `CAP_SYS_ADMIN` is required for `perf_event_open(2)` to run with `docker`.

```
$ head /proc/cpuinfo
processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model           : 94
model name      : Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
stepping        : 3
microcode       : 0x74
cpu MHz         : 1758.593
cache size      : 8192 KB
physical id     : 0
$ docker build --build-arg KERNEL_VERSION=$(uname -r) -t count-dnn-flops .
$ docker run --cap-add SYS_ADMIN -v $PWD:/work -w /work count-dnn-flops python eval_imagenet.py --count-by functions googlenet
"Convolution2DFunction","30507008"
"ReLU","0"
"MaxPooling2D","0"
"LocalResponseNormalization","18919576"
...snip...
"Convolution2DFunction","1315552"
"Concat","0"
"ReLU","0"
"AveragePooling2D","51200"
"Reshape","0"
"LinearFunction","258750"
"Softmax","5995"
```

