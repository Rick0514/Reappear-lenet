from lenet import LeNet
from data import Datagenerator, OutOfRange
import numpy as np
import cv2
import pickle

def get_rbfw_faster(images, labels):
    weights = np.zeros((84,num))
    n, d = labels.shape
    ex = []
    for i in range(n):
        temp = labels[i,:]
        idx = np.argwhere(temp == 1)
        idx = idx[0,0]
        if idx not in ex:
            ex.append(idx)
            img = images[i,0,:,:]
            h, w =img.shape
            pos = [0, 0, w, h]

            for i in range(w):
                s0 = np.sum(img[:,i])
                if s0 == 0:
                    pos[0] += 1
                else:
                    break

            for i in range(w):
                s0 = np.sum(img[:,w-1-i])
                if s0 == 0:
                    pos[2] -= 1
                else:
                    break

            for i in range(h):
                s0 = np.sum(img[i,:])
                if s0 == 0:
                    pos[1] += 1
                else:
                    break

            for i in range(h):
                s0 = np.sum(img[h-1-i,:])
                if s0 == 0:
                    pos[3] -= 1
                else:
                    break

            e0 = pos[2] - pos[0]
            e1 = pos[3] - pos[1]
            if e0 < 7:
                dx = (7 - e0) // 2 + 1
                pos[0] -= dx
                pos[2] += dx

            if e1 < 12:
                dy = (12 - e1) //2 + 1
                pos[1] -= dy
                pos[3] += dy

            newimg = img[pos[1]:pos[3], pos[0]:pos[2]]
            newimg = cv2.resize(newimg, (7,12))

            newimg = newimg.reshape((-1))
            newimg0 = np.ones(newimg.shape)
            newimg0[newimg < 50] = -1
            weights[:,idx] = newimg0

    return weights

# # make Rbfcon weights fast !!!!!!!!!!!!!!!!!
# file = './data'
# num = 10        # 0~9 10 numbers totally
# epochs = 1     # try 10 epochs
# batch_size = 100
# a = Datagenerator(file,'test', num, batch_size)
# a.get_data('batch')
# images, labels = a.gen_batch()
# rbf_w = get_rbfw_faster(images, labels)
#
# with open(r'./rbf_w.pkl', 'wb') as f:
#     pickle.dump(rbf_w, f)


if __name__ == '__main__':

    file = './data'
    num = 10        # 0~9 10 numbers totally
    epochs = 2     # try 10 epochs
    batch_size = 32

    record_loss = []
    record_acc = []

    #get Rbfcon weights
    with open(r'./data/rbf_w.pkl', 'rb') as f:
        rbf_w = pickle.load(f)

    #initialize data generator
    train_generator = Datagenerator(file,'train', num, batch_size)
    test_generator = Datagenerator(file, 'test', num, 200)


    #define and initialize LeNet-5
    net = LeNet(rbf_w)
    net.init_weights()

    step = 0

    for eps in range(epochs):
        train_generator.get_data('batch', True)
        while True:
            try:
                train_images, train_labels = train_generator.gen_batch()

                train_loss, train_acc = net.train(train_images, train_labels)

                if step % 10 == 0 :
                    record_loss.append(train_loss)
                    record_acc.append(train_acc)
                    print('step --> %d, loss : %.4f, acc : %.2f' % (step, train_loss, train_acc))

                step += 1
            except OutOfRange:
                # finish a epoch, validate test set
                test_generator.get_data('batch', False)
                tacc = []
                while True:
                    try:
                        test_images, test_labels = test_generator.gen_batch()
                        test_acc = net.test(test_images, test_labels)
                        tacc.append(test_acc)
                    except OutOfRange:
                        tacc = np.array(tacc).mean()
                        print('%d epochs --> acc : %.2f ' % (eps, tacc))
                        break

                break

    np.savez('./record.npz', record_loss=record_loss, record_acc=record_acc)
