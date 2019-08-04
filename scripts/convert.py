#!/usr/bin/env python3

import mxnet as mx
import numpy as np
import torch
from imageio import imread

import insightface.iresnet as models

archs = ['iresnet34', 'iresnet50', 'iresnet100']
imsize = 112


def convert(arch):
    path = 'resource/{arch}/model'.format(arch=arch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)
    params = arg_params.copy()
    params.update(aux_params)

    state_dict = dict()
    for k, v in sorted(params.items()):
        k = k.replace('conv0', 'conv1')

        # stage1_unit1 -> layer1.0
        k = k.replace('_', '.')
        k = k.replace('stage', 'layer')
        if 'unit' in k:
            k = k.replace('unit', '')
            k = '.'.join(k.split('.')[:1]
                         + [str(int(k.split('.')[1]) - 1)]
                         + k.split('.')[2:])

        # relu / bn
        if k.startswith('bn1'):
            k = k.replace('bn1', 'bn2')
        if k.startswith('bn0'):
            k = k.replace('bn0', 'bn1')
        k = k.replace('relu0', 'prelu')
        k = k.replace('relu1', 'prelu')
        k = k.replace('beta', 'bias')
        k = k.replace('gamma', 'weight')
        k = k.replace('moving', 'running')
        k = k.replace('running.', 'running_')

        # residual fix
        k = k.replace('.conv1sc', '.downsample.0')
        k = k.replace('.sc', '.downsample.1')

        k = k.replace('pre.fc1', 'fc')
        k = k.replace('fc1', 'features')

        state_dict[k] = torch.from_numpy(v.asnumpy())

    # original model
    ctx = mx.cpu()
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    mxmod = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    mxmod.bind(data_shapes=[('data', (1, 3, imsize, imsize))])
    mxmod.set_params(arg_params, aux_params)

    # converted model
    sota = getattr(models, arch)()
    sota.load_state_dict(state_dict)
    sota.eval()

    torch.set_grad_enabled(False)

    # random image
    image = imread('resource/sample.jpg')
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)

    # running original model
    mx_tensor = mx.io.DataBatch(data=(mx.nd.array(image),))
    mxmod.forward(mx_tensor, is_train=False)
    mx_output = mxmod.get_outputs()[0].asnumpy().flatten()

    # running converted model
    th_tensor = (torch.from_numpy(image).float() - 127.5) / 128.0
    th_output = sota(th_tensor).numpy().flatten()

    # check
    print('mx', mx_output[:5])
    print('th', th_output[:5])

    assert np.allclose(mx_output, th_output, atol=1e-5)

    torch.save(sota.state_dict(), 'resource/{arch}.pth'.format(arch=arch))


def main():
    for arch in archs:
        convert(arch)


if __name__ == '__main__':
    main()
