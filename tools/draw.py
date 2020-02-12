import matplotlib.pyplot as plt
import numpy as np

record = np.load(r'./record.npz')
loss = record['record_loss']
acc = record['record_acc']

x = np.arange(len(loss)) * 10

plt.subplot(121)
plt.plot(x, loss)
plt.xlabel('step')
plt.ylabel('loss')
plt.title('2-epochs')
plt.grid()

plt.subplot(122)
plt.plot(x, acc)
plt.title('2-epochs')
plt.xlabel('step')
plt.ylabel('accuracy')
plt.grid()