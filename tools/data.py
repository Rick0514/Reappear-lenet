import os
import numpy as np
import struct

class OutOfRange(Exception):
    pass

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def preprocesssing(images):
    #n x dim(28*28)
    # mu = np.mean(images)
    # std = np.std(images)
    # return (images - mu) / std
    return (images - 128.0) / 128.0

class Datagenerator:

    def __init__(self, filename, kind, num, batch_size):

        self.file = filename
        self.batch_size = batch_size
        self.num = num
        images, labels = load_mnist(self.file, kind)
        images = preprocesssing(images)
        self.data = [images, labels]


    def get_data(self, mode, shuffle=True):
        #just for MNIST dataset, for test, as the generator assume lots of memory

        data_id = np.arange(len(self.data[1]))
        if shuffle:
            np.random.shuffle(data_id)

        data_id = list(data_id)

        if mode == 'batch':
            num = len(data_id)
            temp = []
            cnt = 0
            while cnt+self.batch_size < num:
                temp.append(data_id[cnt:cnt+self.batch_size])
                cnt += self.batch_size
            temp.append(data_id[cnt:])
            data_id = temp
        else:
            data_id = [data_id]

        #[batch,[idx]]
        self.data_id = data_id
        self.cnt = 0

    def gen_batch(self):
        if self.cnt >= len(self.data_id):
            raise OutOfRange
        else:
            data_id = self.data_id[self.cnt]
            images = self.data[0][data_id,:]

            temp0 = [images[i,:].reshape((1,28,28)) for i in range(images.shape[0])]

            images = np.stack(temp0, axis=0)    #nx1x28x28
            labels = self.data[1][data_id]
            labels0 = np.zeros((labels.shape[0], self.num))
            labels0[np.arange(labels.shape[0]),labels] = 1       #  one-hot encoded
            labels = labels0

        self.cnt +=1

        return images, labels


# a = Datagenerator(r'./data', 'train', 10, 1000)
# a.get_data('batch', False)
#
# while True:
#     try:
#         b, c = a.gen_batch()
#     except OutOfRange:
#         break
