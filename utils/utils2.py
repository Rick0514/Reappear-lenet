import numpy as np

# version2.0 accelarate convolution speed with im2col method
# fix some error in version0.0
# (e.g in reconv there are some unknown error to be fixed,
# in repool has some unknown error)


def im2col(inputs, filter_size, stride=1, pad=(0, 0)):
    N, C, H, W = inputs.shape
    FH, FW = filter_size

    out_h = (H + 2 * pad[0] - FH) // stride + 1
    out_w = (W + 2 * pad[1] - FW) // stride + 1

    img = np.pad(inputs, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), 'constant')

    col = np.zeros((N, C, FH, FW, out_h, out_w))

    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col, out_h, out_w

def col2im(col, input_shape, filter_shape, out_shape, stride=1, pad=(0,0)):
    N, C, H, W = input_shape
    out_h ,out_w = out_shape
    filter_h, filter_w = filter_shape

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad[0] + stride - 1, W + 2*pad[1] + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad[0]:H + pad[0], pad[1]:W + pad[1]]


def conv2(inputs, kernel, bias, stride=1, pad=(0, 0)):
    N, C, H, W = inputs.shape
    FN, FC, FH, FW = kernel.shape

    # [NxOHxOW, CxFHxFW]
    col, OH, OW = im2col(inputs, [FH, FW], stride, pad)
    ker = kernel.reshape((FN, -1)).T  # [FCxFHxFW, FN]

    dot = np.dot(col, ker)  # [NxOHxOW, FN]
    # bias [1, FN]
    result = dot + bias

    result = result.T.reshape((FN, N, OH, OW))
    result = result.transpose(1, 0, 2, 3)

    return result


def flip180(arr):
    FC, FN, FH, FW = arr.shape
    new_arr = arr.reshape((FC, FN, -1))
    new_arr = new_arr[..., ::-1]
    new_arr = new_arr.reshape((FC, FN, FH, FW))
    return new_arr


def reconv2(delta_in, kernel, stride=1, pad=(0, 0)):
    N, FN, OH, OW = delta_in.shape
    FN, FC, FH, FW = kernel.shape

    kernel0 = kernel.transpose(1, 0, 2, 3)  # FCxFNxFHxFW
    kernel0 = flip180(kernel0)

    if stride > 1:
        hid = np.repeat(np.arange(1,OH), stride-1)
        wid = np.repeat(np.arange(1,OW), stride-1)
        delta_in = np.insert(delta_in, hid, 0, axis=2)
        delta_in = np.insert(delta_in, wid, 0, axis=3)

    delta_out = conv2(delta_in, kernel0, 0, pad=(FH-1, FW-1))

    N, C, H1, W1 = delta_out.shape
    delta_out = delta_out[..., pad[0]:H1 - pad[0], pad[1]:W1 - pad[1]]

    return delta_out

def pool2(inputs, kernel_size, kind, stride=1, pad=(0,0)):
    N, C, H, W = inputs.shape
    FH, FW = kernel_size[:2]

    if kind == 'mean':
        col, out_h, out_w = im2col(inputs, (FH, FW), stride, pad)
        col = col.reshape((-1, FH * FW))
        ccol = 1.0 * np.ones(col.shape) / (FH * FW)
        col = np.mean(col, axis=1)
        col = col.reshape((N,out_h,out_w,C)).transpose(0,3,1,2)

    elif kind == 'max':
        col, out_h, out_w = im2col(inputs, (FH, FW), stride, pad)
        col = col.reshape((-1, FH*FW))
        col0 = np.max(col, axis=1, keepdims=True)
        ccol = 1.0 * (col == col0)
        col = col0.reshape((N, out_h, out_w, C)).transpose(0,3,1,2)

    out_shape = (out_h, out_w)
    return col, ccol, out_shape   #ccol [NxOHxOWxC, FHxFW]

def repool2(delta_in, ccol, inputs_shape, kernel_shape, out_shape, stride=1, pad=(0,0)):
    N, C, OH, OW = delta_in.shape

    delta_in = delta_in.transpose(0,2,3,1)  # NxOHxOWxC
    delta_in = delta_in.reshape((delta_in.size, 1))    # [NxOHxOWxC, 1]

    delta_out = ccol * delta_in
    delta_out = delta_out.reshape((N*OH*OW, -1))

    delta_out = col2im(delta_out, inputs_shape, kernel_shape, out_shape, stride, pad)

    return delta_out

def activate(inputs, kind):
    inputs = np.clip(inputs, -1e2, 1e2)
    if kind == 'sigmoid':
        outputs = 1.0 / (1 + np.exp(0 - inputs))
    elif kind == 'relu':
        outputs = np.maximum(inputs, 0)

    elif kind == 'Letanh':
        A = 1.7159
        S = 2.0 / 3
        t0 = np.exp(S * inputs)
        t1 = 1.0 / t0
        outputs = A * (t0 - t1) / (t0 + t1)
    else:
        outputs = inputs

    return outputs

def deactivate(inputs, kind):
    inputs = np.clip(inputs, -1e2, 1e2)
    if kind == 'sigmoid':
        temp = 1 / (1 + np.exp(0 - inputs))
        outputs = temp * (1 - temp)

    elif kind == 'relu':
        outputs = 1.0 * (inputs > 0)

    elif kind == 'Letanh':
        A = 1.7159
        S = 2.0 / 3
        t0 = np.exp(S * inputs)
        t1 = 1.0 / t0
        temp = (t0 - t1) / (t0 + t1)
        outputs = A * S * (1.0 - temp ** 2)
    else:
        return 1.0

    return outputs


#
# inputs = np.stack((np.arange(0, 9), np.arange(9,18)), axis=0)
# inputs = inputs.reshape((2, 3, 3))
# inputs = np.stack((inputs, inputs), axis=0)  # 2x2x3x3
#
# inputs1 = inputs.transpose(0, 2, 3, 1)

#
# inputs1 = np.stack((np.arange(0,625), np.arange(0,625)), axis=0)
# inputs1 = inputs.reshape((2,1,25,25))
#
# kernel = np.array([[1, 0], [0, 0]])
# kernel = np.stack((kernel, kernel), axis=0)  # 2x2x2
# kernel = np.stack((kernel, 2 * kernel, 3 * kernel), axis=0)  # 3x2x2x2
# kernel1 = kernel.transpose(0, 2, 3, 1)


# num = 10
#
# t0 = time.time()
# for i in range(num):
#     a, aa, out_shape = pool2(inputs, [2,2], 'mean')
#     b = repool2(a, aa, inputs.shape, [2, 2], out_shape)
#
# t1 = time.time()
# print('time cost: %f s' % (t1 - t0))
#
#
# t0 = time.time()
# for i in range(num):
#     a, aa = pool(inputs1, [2,2,2], [1,1], 'mean', 'valid')
#     bb = repool(a, aa, [2,2,2], [1,1])
# t1 = time.time()
# print('time cost: %f s' % (t1 - t0))



# kernel = np.expand_dims(kernel, axis=3)
#
# kernel1 = kernel.reshape((2,1,2,2))
# bias = np.zeros((1,2))
#
#
# num = 10

# t0 = time.time()
# for i in range(num):
#     a = conv(inputs, kernel, [1,1], 'valid')
# # 2x2x2x2
# t1 = time.time()
# print('time cost: %f s' % (t1-t0))
#
# t0 = time.time()
# for i in range(num):
#     b = conv2(inputs1, kernel1, bias)
# t1 = time.time()
# print('time cost: %f s' % (t1-t0))

# t0 = time.time()
# a = conv(inputs, kernel, [1,1], 'valid')
# t1 = time.time()
# print('time cost: %f s' % (t1-t0))
#
#
# t0 = time.time()
# for i in range(num):
#     b = conv2(inputs1, kernel1, bias)
# t1 = time.time()
# print('time cost: %f s' % (t1-t0))
# # # #
# # a = reconv(inputs, [1,4,4,1], [2,2], kernel)
#
# a, out_a = pool(inputs, [2,2], [1,1], 'max', 'valid')
#
# b = repool(a, out_a, [2,2], [1,1])
#
# print(b[0,:,:,0])
# print(b[0,:,:,1])

# import tensorflow as tf
#
# inp = tf.placeholder(tf.float16, [2,25,25,1])
# inputs2 = inputs.astype(np.float16)
# inp = tf.Variable(inputs2)
# kernel2 = kernel.transpose(1,2,3,0)
# kernel2 = kernel2.astype(np.float16)
# w = tf.Variable(kernel2, name='w')
# r = tf.nn.conv2d(inp, w, strides=[1,1,1,1], padding='VALID')
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     t0 = time.time()
#     for i in range(num):
#         rr = sess.run(r)
#     t1 = time.time()
#     print('time cost: %f s' % (t1 - t0))