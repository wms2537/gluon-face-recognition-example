# import the necessary packages
from imutils import paths
import argparse
import pickle
import cv2
import os

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import mxnet.autograd as ag
from mxnet import gluon
from mxnet.gluon import nn
from gluoncv.data.batchify import Stack, Tuple
from gluoncv.utils import LRScheduler, LRSequential
import numpy as np
import gluoncv as gcv
from tinydb import TinyDB
from tqdm import tqdm


class NormDense(nn.HybridBlock):
    """Norm Dense"""
    def __init__(self, classes, weight_norm=False, feature_norm=False,
                 dtype='float32', weight_initializer=None, in_units=0, **kwargs):
        super().__init__(**kwargs)
        self._weight_norm = weight_norm
        self._feature_norm = feature_norm

        self._classes = classes
        self._in_units = in_units
        if weight_norm:
            assert in_units > 0, "Weight shape cannot be inferred auto when use weight norm, " \
                                 "in_units should be given."
        self.weight = gluon.Parameter('weight', shape=(classes, in_units),
                                        init=weight_initializer, dtype=dtype,
                                        allow_deferred_init=True)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x, weight, *args, **kwargs):
        if self._weight_norm:
            weight = F.L2Normalization(weight, mode='instance')
        if self._feature_norm:
            x = F.L2Normalization(x, mode='instance', name='fc1n')
        return F.FullyConnected(data=x, weight=weight, no_bias=True,
                                num_hidden=self._classes, name='fc7')

    def __repr__(self):
        s = '{name}({layout})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


db = TinyDB('./d_single_image_ids.json')
db.truncate()

factor = 0.02 * 512 # loosely crop face
sample_size = 5
ctx = [mx.gpu(0)]
det_net = gluon.nn.SymbolBlock.imports("./models/face_detection/center_net_resnet18_v1b_face_best-symbol.json", [
                                          'data'], "./models/face_detection/center_net_resnet18_v1b_face_best-0113.params", ctx=ctx) # face detection

features_net = mx.gluon.nn.SymbolBlock.imports(
    "./models/face_feature_extraction/mobilefacenet-symbol.json", ['data'], "./models/face_feature_extraction/mobilefacenet-0000.params", ctx=ctx) # face feature extraction

batch_size = 32
num_samples = 907
num_classes = 907
face_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomBrightness(0.3),
    transforms.RandomContrast(0.3),
    transforms.RandomSaturation(0.3),
    transforms.ToTensor(),
    transforms.Cast('float32')
])

dataset_train = gluon.data.vision.datasets.ImageFolderDataset(
    './singleimagedataset2')
db.insert({'labels': dataset_train.synsets})

train_data = gluon.data.DataLoader(dataset_train.transform_first(face_transforms), batch_size=batch_size, shuffle=True, num_workers=6, batchify_fn=Tuple(Stack(), Stack()), last_batch='rollover')

net = nn.HybridSequential()


net.add(nn.Dense(512, in_units=512))   # input layer
net.add(NormDense(num_classes, True, True,
                  in_units=512))   # output layer
net.hybridize(static_alloc=True, static_shape=True)
net.initialize(mx.init.Xavier(), ctx=ctx)

lossfn = gluon.loss.SoftmaxCrossEntropyLoss()
epochs = 30
warmup_epochs = 0
lr = 1e-3
num_batches = num_samples // batch_size
lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=0, target_lr=lr,
                nepochs=warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler("cosine", base_lr=lr, target_lr=1e-7,
                nepochs=epochs - warmup_epochs,
                iters_per_epoch=num_batches),
])

optimizer = mx.optimizer.Adam(wd=1e-4, lr_scheduler=lr_scheduler)
trainer = gluon.Trainer(net.collect_params(), optimizer)
loss_mtc = gluon.metric.Loss()
top1_acc = gluon.metric.Accuracy()
top5_acc = gluon.metric.TopKAccuracy(top_k=5)

for epoch in range(epochs):
    pbar = tqdm(train_data)
    for (i, batch) in enumerate(pbar):
        datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        labels = gluon.utils.split_and_load(
            batch[1], ctx_list=ctx, batch_axis=0)
        features = [features_net(d) for d in datas]
        outputs = None
        with ag.record():
            outputs = [net(X) for X in features]
            # print(datas, labels)
            losses = [lossfn(yhat, y) for yhat, y in zip(outputs, labels)]
            ag.backward(losses)
        trainer.step(batch_size)
        top1_acc.update(labels, outputs)
        top5_acc.update(labels, outputs)
        loss_mtc.update(..., losses)
        _, top1 = top1_acc.get()
        _, top5 = top5_acc.get()
        _, loss_val = loss_mtc.get()
        pbar.set_postfix(epoch=epoch, top1=top1, top5=top5, loss=loss_val)
    _, top1 = top1_acc.get()
    _, top5 = top5_acc.get()
    print('[Epoch %d] top1=%f top5=%f'%
          (epoch, top1, top5))
    net.export('./models/mlp', epoch=epoch)


        
