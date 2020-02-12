from utils2 import *

class Conv2:
    def __init__(self, kernel_size, stride=1, pad=(0,0)):
        self.pad = pad
        self.stride = stride
        self.kernel_size = kernel_size

    def init_weights(self, init_kind):
        FN, FC, FH, FW = self.kernel_size
        # lenet-5 C1 and C3 have no activation, therefor init_kind may choose randomly
        if init_kind == 'Gaussian':
            std = 0.01
            kernel = np.random.normal(0, std, (FN, FC, FH, FW))
            bias = np.random.normal(0, 0.01, (1, FN))
        elif init_kind == 'He':  # for relu activation
            std = np.sqrt(2 / (FN*FC*FH*FW))
            kernel = np.random.normal(0, std, (FN, FC, FH, FW))
            bias = np.random.normal(0, 0.01, (1, FN))
        elif init_kind == 'Le': #such initialation method is applied in the lenet-5
            fin = FC*FH*FW		#according to the essay
            low = -2.4 / fin
            high = -low
            kernel = np.random.uniform(low, high, (FN, FC, FH, FW))
            bias = np.random.uniform(low, high ,(1, FN))

        self.kernel = kernel
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = conv2(self.inputs, self.kernel, self.bias, self.stride, self.pad)

    def update(self, delta_in, learning_rate):
        # [NxFNxOHxOW]
        if len(delta_in.shape) < 4:
            delta_in = np.expand_dims(delta_in, axis=(2,3))
        self.delta_in = delta_in

        self.delta_out = reconv2(self.delta_in, self.kernel, self.stride, self.pad)

        # [1, FN]
        temp = np.sum(self.delta_in, axis=(0,2,3))
        temp = temp.reshape((1,-1))
        self.bias -= learning_rate * temp

        delta_in0 = self.delta_in.swapaxes(0,1) #[FNxNxOHxOW]
        inputs0 = self.inputs.swapaxes(0,1)     #[CxNxHxW]
        kernel_gra = conv2(inputs0, delta_in0, 0, self.stride, self.pad)    #[CxFNxFHxFW]
        kernel_gra = kernel_gra.swapaxes(0,1)
        self.kernel = self.kernel - learning_rate * kernel_gra


class Conv:
    def __init__(self, kernel_size, strides, padding):

        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size

    def init_weights(self, init_kind):
        kn, kh, kw, kd = self.kernel_size
        # lenet-5 C1 and C3 have no activation, therefor init_kind may choose randomly
        if init_kind == 'Gaussian':
            std = 0.01
            kernel = np.random.normal(0, std, (kn, kh, kw, kd))
            bias = np.random.normal(0, 0.01, (1, 1, 1, kn))
        elif init_kind == 'He':  # for relu activation
            std = np.sqrt(2 / (kn * kh * kw * kd))
            kernel = np.random.normal(0, std, (kn, kh, kw, kd))
            bias = np.random.normal(0, 0.01, (1, 1, 1, kn))
        else:  # for test
            kernel = np.ones((kn, kh, kw, kd))
            bias = np.zeros((1, 1, 1, kn))

        self.kernel = kernel
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        outputs = conv(self.inputs, self.kernel, self.strides, self.padding) + self.bias
        self.outputs = outputs

    def update(self, delta_in, learning_rate):

        self.delta_in = delta_in
        ni, hi, wi, di = self.inputs.shape

        delta_out = reconv(self.delta_in, [ni, hi, wi, di], self.strides, self.kernel)
        self.delta_out = delta_out

        bias_gra = self.delta_in
        self.bias = self.bias - learning_rate * np.sum(bias_gra, axis=(0, 1, 2), keepdims=True)

        kernel_gra = conv_kernel(self.inputs, bias_gra, self.strides, self.padding)

        self.kernel = self.kernel - learning_rate * kernel_gra

class Leconv2:
    def __init__(self):
        pos = np.zeros((6, 16))
        temp0 = [np.array([0, 1, 2]), np.array([0, 1, 2, 3]),
                 np.array([0, 1, 3, 4]), np.arange(6)]
        j = 0
        for i in range(16):
            if i == 6 or i == 12 or i == 15:
                j += 1
            pos[temp0[j], i] = 1
            temp0[j] += 1
            temp0[j][temp0[j] == 6] = 0

        self.pos = pos.astype(np.bool)

        block = []
        k_size = [(1, 3, 5, 5), (1, 4, 5, 5), (1, 6, 5, 5)]
        j = 0
        for i in range(16):
            if i == 6 or i == 15:
                j += 1
            block.append(Conv2(k_size[j], stride=1, pad=(0,0)))
        self.block = block

    def init_weights(self, init_kind):
        for each in self.block:
            each.init_weights(init_kind)

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []
        for k, each in enumerate(self.block):
            inputs0 = inputs[:,self.pos[:, k],...]
            each.forward(inputs0)
            outputs.append(each.outputs)

        self.outputs = np.concatenate(outputs, axis=1)

    def update(self, delta_in, learning_rate):

        self.delta_in = delta_in    # Nx16xOHxOW
        self.delta_out = np.zeros(self.inputs.shape)
        for k, each in enumerate(self.block):
            delta = delta_in[:,k,...]
            delta = np.expand_dims(delta, axis=1)
            each.update(delta, learning_rate)
            self.delta_out[:,self.pos[:, k],:,:] += each.delta_out

class Leconv:
    def __init__(self):
        pos = np.zeros((6, 16))
        temp0 = [np.array([0, 1, 2]), np.array([0, 1, 2, 3]),
                 np.array([0, 1, 3, 4]), np.arange(6)]
        j = 0
        for i in range(16):
            if i == 6 or i == 12 or i == 15:
                j += 1
            pos[temp0[j], i] = 1
            temp0[j] += 1
            temp0[j][temp0[j] == 6] = 0

        self.pos = pos.astype(np.bool)

        block = []
        k_size = [(1, 5, 5, 3), (1, 5, 5, 4), (1, 5, 5, 6)]
        j = 0
        for i in range(16):
            if i == 6 or i == 15:
                j += 1
            block.append(Conv(k_size[j], [1, 1], 'valid'))
        self.block = block

    def init_weights(self, init_kind):
        for each in self.block:
            each.init_weights(init_kind)

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []
        for k, each in enumerate(self.block):
            inputs0 = inputs[..., self.pos[:, k]]
            each.forward(inputs0)
            outputs.append(each.outputs)

        self.outputs = np.concatenate(outputs, axis=3)

    def update(self, delta_in, learning_rate):
        self.delta_in = delta_in
        self.delta_out = np.zeros(self.inputs.shape)
        for k, each in enumerate(self.block):
            delta = delta_in[..., k]
            delta = np.expand_dims(delta, axis=3)
            each.update(delta, learning_rate)
            self.delta_out[..., self.pos[:, k]] += each.delta_out

class Pool2:
    def __init__(self, kernel_size, kind, act_kind, stride, pad=(0,0)):

        self.kind = kind
        self.pad = pad
        self.stride = stride
        self.kernel_size = kernel_size  #[h, w, d]
        self.act_kind = act_kind

    def init_weights(self, init_kind):
        FH, FW, n = self.kernel_size
        if init_kind == 'Gaussian':
            std = 0.01
            beta = np.random.normal(0, std, (1, n))
            bias = np.random.normal(0, 0.01, (1, n))  # ??
        elif init_kind == 'He':  # for relu activation
            std = np.sqrt(2 / n)
            beta = np.random.normal(0, std, (1, n))
            bias = np.random.normal(0, 0.01, (1, n))  # ??
        elif init_kind == 'Le':
            fin = n
            low = -2.4 / fin
            high = -low
            beta = np.random.uniform(low, high, (1,n))
            bias = np.random.uniform(low, high ,(1, n))

        self.beta = beta
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        self.outputu, self.output0, self.out_shape = pool2(self.inputs,
                        self.kernel_size, self.kind, self.stride, self.pad)

        beta = np.expand_dims(self.beta, axis=(2,3))
        bias = np.expand_dims(self.bias, axis=(2,3))
        outputs = beta * self.outputu + bias
        self.outputu = outputs
        self.outputs = activate(outputs, self.act_kind)

    def update(self, delta_in, learning_rate):

        self.delta_in = delta_in * deactivate(self.outputu, self.act_kind)

        delta_out = repool2(self.delta_in, self.output0, self.inputs.shape,
                            self.kernel_size[:2], self.out_shape, self.stride, self.pad)
        # [NxCxHxW]
        beta = np.expand_dims(self.beta, axis=(2, 3))
        self.delta_out = beta * delta_out

        temp = np.sum(np.sum(self.delta_in, axis=(2,3)), axis=0, keepdims=True)
        self.bias = self.bias - learning_rate * temp
        temp = np.sum(np.sum(self.delta_in * self.outputu, axis=(2, 3)), axis=0, keepdims=True)
        self.beta = self.beta - learning_rate * temp


class Pool:
    def __init__(self, kernel_size, strides, kind, padding, act_kind):

        self.kind = kind
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size

        self.act_kind = act_kind

    def init_weights(self, init_kind):
        n = self.kernel_size[2]
        if init_kind == 'Gaussian':
            std = 0.01
            beta = np.random.normal(0, std, (1, n))
            bias = np.random.normal(0, 0.01, (1, n))  # ??
        elif init_kind == 'He':  # for relu activation
            std = np.sqrt(2 / n)
            beta = np.random.normal(0, std, (1, n))
            bias = np.random.normal(0, 0.01, (1, n))  # ??
        else:
            beta = np.ones((1, n))
            bias = np.zeros((1, n))

        self.beta = beta
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        n = self.inputs.shape[3]
        self.outputu, self.output0 = pool(self.inputs, self.kernel_size, self.strides, self.kind, self.padding)
        beta = self.beta.reshape((1, 1, 1, n))
        bias = self.bias.reshape((1, 1, 1, n))
        outputs = beta * self.outputu + bias
        self.outputu = outputs
        self.outputs = activate(outputs, self.act_kind)

    def update(self, delta_in, learning_rate):
        self.delta_in = delta_in

        self.delta_in = self.delta_in * deactivate(self.outputu, self.act_kind)

        delta_out = repool(self.delta_in, self.output0, self.kernel_size, self.strides)
        self.delta_out = self.beta * delta_out

        temp = np.sum(np.sum(self.delta_in, axis=(1, 2)), axis=0, keepdims=True)
        self.bias = self.bias - learning_rate * temp
        temp = np.sum(np.sum(self.delta_in * self.outputu, axis=(1, 2)), axis=0, keepdims=True)
        self.beta = self.beta - learning_rate * temp


class Fullycon:
    def __init__(self, num, act_kind):

        self.num = num
        self.act_kind = act_kind

    def init_weights(self, init_kind):

        c1, c2 = self.num

        if init_kind == 'Xavier':  # use when activates with tanh
            std = np.sqrt(1 / c1)
            self.weights = np.random.normal(0, std, (c1, c2))
            self.bias = np.random.normal(0, 0.01, (1, c2))
        elif init_kind == 'He':  # use when activates with relu
            std = np.sqrt(2 / c1)
            self.weights = np.random.normal(0, std, (c1, c2))
            self.bias = np.random.normal(0, 0.01, (1, c2))
        else:
            self.weights = np.ones((c1, c2))
            self.bias = np.ones((1, c2))

    def forward(self, inputs):
        self.inputs = inputs
        if len(self.inputs.shape) > 2:
            self.inputs = np.squeeze(self.inputs, (1, 2))
        outputu = np.matmul(self.inputs, self.weights) + self.bias
        self.outputu = outputu
        self.outputs = activate(outputu, self.act_kind)

    def update(self, delta_in, learning_rate):
        self.delta_in = delta_in
        self.delta_in = self.delta_in * deactivate(self.outputu, self.act_kind)
        if len(self.delta_in.shape) > 2:
            self.delta_in = np.squeeze(self.delta_in, (1, 2))

        inputs = self.inputs
        temp = np.matmul(self.weights, self.delta_in.T)
        delta_out = temp.T
        self.delta_out = np.expand_dims(delta_out, (1, 2))

        self.bias = self.bias - learning_rate * self.delta_in
        self.weights = self.weights - learning_rate * np.matmul(inputs.T, self.delta_in)

class Fullycon2:
    def __init__(self, num, act_kind):

        self.num = num
        self.act_kind = act_kind

    def init_weights(self, init_kind):

        c1, c2 = self.num
        if init_kind == 'Gaussian':
            std = 0.01
            self.weights = np.random.normal(0, std, (c1, c2))
            self.bias = np.random.normal(0, 0.01, (1, c2))  # ??
        elif init_kind == 'Xavier':  # use when activates with tanh
            std = np.sqrt(2 / (c1 + c2))
            self.weights = np.random.normal(0, std, (c1, c2))
            self.bias = np.random.normal(0, 0.01, (1, c2))
        elif init_kind == 'He':  # use when activates with relu
            std = np.sqrt(2 / c1)
            self.weights = np.random.normal(0, std, (c1, c2))
            self.bias = np.random.normal(0, 0.01, (1, c2))
        else:
            self.weights = np.ones((c1, c2))
            self.bias = np.ones((1, c2))

        return self.weights, self.bias

    def forward(self, inputs):
        if len(inputs.shape) > 2:   #NxCxHxW mx120x1x1
            inputs = np.squeeze(inputs, axis=(2,3)) #mx120
        self.inputs = inputs
        self.outputu = np.matmul(self.inputs, self.weights) + self.bias
        self.outputs = activate(self.outputu, self.act_kind)

    def update(self, delta_in, learning_rate):
        #Nxc2
        self.delta_in = delta_in * deactivate(self.outputu, self.act_kind)

        inputs = self.inputs    #Nxc1
        self.delta_out = np.matmul(self.delta_in, self.weights.T) #Nxc2 x c2xc1

        bias_gra = np.sum(self.delta_in, axis=0, keepdims=True)
        self.bias -= learning_rate * bias_gra
        ker_gra = np.matmul(inputs.T, self.delta_in)
        self.weights -= learning_rate * ker_gra

        return ker_gra, bias_gra

class Rbfcon2:

    def __init__(self, weights):

        self.weights = weights

    def forward(self, inputs):
        self.inputs = inputs  # num * dim
        n, d = self.inputs.shape
        # n_in, n_out = self.weights.shape

        temp0 = np.expand_dims(inputs, axis=2)  #nx84x1
        weights = np.expand_dims(self.weights, axis=0).repeat(n, axis=0)    #nx84x10

        temp0 = (temp0 - weights) ** 2
        self.outputs = temp0.sum(axis=1)    # num * 10


    def update(self, delta_in, learning_rate):
        self.delta_in = delta_in    #mx10
        n, d = self.inputs.shape

        inputs = np.expand_dims(self.inputs, axis=2)    # mx84x1
        weights = np.expand_dims(self.weights, axis=0).repeat(n, axis=0)    #mx84x10
        delta = np.expand_dims(delta_in, axis=1)    #mx1x10

        delta_out = 2 * (inputs - weights) * delta   # mx84x10 x mx1x10
        kernel_gra = np.sum(-1.0 * delta_out, axis=0)   #84x10

        self.delta_out = np.sum(delta_out, axis=-1) # mx84
        self.weights = self.weights - learning_rate * kernel_gra


def Lossfun(outputs, labels, kind):
    # outputs nxd
    # labels  nxd
    n, d = labels.shape
    if kind == 'MSE':
        y0 = np.sum(outputs, axis=-1)
        y = outputs / y0
        temp = y - labels
        loss = np.sum(temp ** 2) / (2*n)
        delta_in = 2 * temp * (1 - y) / y0 / (2*n)
    elif kind == 'CE':
        # softmax
        exp_o = np.exp(outputs)
        exp_o1 = np.sum(exp_o, axis=-1, keepdims=True)
        softmax = exp_o / exp_o1
        softmax = np.clip(softmax, 1e-10, 1)
        # outputs0[outputs0 == 0] = 1e-10
        ce = -1.0 * np.log(softmax) * labels
        loss = np.sum(ce) / n
        delta_in = softmax - labels
    elif kind == 'LeLoss':
        j = np.exp(-10)
        e = 1.0
        loss0 = np.sum(outputs * labels)
        loss = loss0 / n
        delta_in = 1.0 * labels
        # exp_o0 = np.exp(-1.0 * outputs)
        # exp_o1 = np.sum(exp_o0, axis=-1, keepdims=True)
        # exp_o2 = np.log(j + exp_o1)
        # loss = (loss0 + e * np.sum(exp_o2)) / n
        # delta_in = (labels - e * exp_o0 / (exp_o1 + j))

    return loss, delta_in


def get_accuracy(outputs, labels):
    n, d = outputs.shape
    max_id = np.argmax(outputs, axis=1)
    pre = np.zeros(outputs.shape)
    pre[np.arange(n), max_id] = 1

    acc = pre * labels
    acc = np.sum(acc) / n

    return acc

def get_accuracy_lenet(outputs, labels):
    n, d = outputs.shape
    min_id = np.argmin(outputs, axis=1)
    pre = np.zeros(outputs.shape)
    pre[np.arange(n), min_id] = 1

    acc = pre * labels
    acc = np.sum(acc) / n

    return acc

# inputs = np.stack((np.arange(0,16), np.arange(16,32)), axis=1)
# inputs = inputs.reshape((4,4,2))
# inputs = np.stack((inputs,inputs), axis=0)  #2x4x4x2

# a = np.arange(4)
# a = a.reshape((2,2,1))

# a = Leconv()

# inputs = np.arange(6)
# inputs = inputs.reshape((2,3))
# weights = np.ones((3,2))

# kernel = np.array([[1,0],[0,0]])
# kernel = np.stack((kernel, kernel), axis=2)
# kernel = np.expand_dims(kernel, (0))  #1x2x2x2
# kernel = np.stack((kernel, kernel, kernel), axis=0)

# inputs = np.arange(5)
# inputs = np.reshape(inputs, (1,1,1,5))
# delta_in = np.ones((2,2))
#
# a = Rbfcon(inputs, delta_in, weights)
#
# # a.init_weights('test')
#
# a.forward()
#
# a.update(1)
