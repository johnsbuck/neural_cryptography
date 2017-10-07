import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from parity_machine import TreeParityMachine
from trainers import tpm_trainer


progress = []
lengths = []
n_examples = 10000
for ex in xrange(n_examples):
    print ex
    model_a = TreeParityMachine(10, 10, 3)
    model_b = TreeParityMachine(10, 10, 3)
    eve = TreeParityMachine(10, 10, 3)

    data = tpm_trainer.train(model_a, model_b, eve, print_step=1000)
    progress.append(np.array(data[-1]))
    lengths.append(data[1])

high_length = 0
low_length = -1
avg_length = 0
for prog in progress:
    if prog.shape[0] > high_length:
        high_length = prog.shape[0]
    elif prog.shape[0] < low_length or low_length < 0:
        low_length = prog.shape[0]
    avg_length += prog.shape[0]
avg_length /= 1. * len(progress)

prob, x = np.histogram(lengths, bins=n_examples/100)
x = x[:-1] + (x[1] - x[0])/2.
f = UnivariateSpline(x, prob, s=n_examples/100)
plt.plot(x, f(x))
plt.show()

print 'Statistics'
print '~~~~~~~~~~'
print 'Highest Length :', high_length
print 'Lowest Length  :', low_length
print 'Average Length :', avg_length

for i in xrange(0, len(progress)):
    plt.plot(progress[i][:, 0], label="A vs B (" + str(i) + ")")
    plt.plot(progress[i][:, 1], label="A vs Eve (" + str(i) + ")")
    plt.plot(progress[i][:, 2], label="B vs Eve (" + str(i) + ")")

    plt.xlabel('Number of Iterations')
    plt.ylabel('Percent Match')
    plt.title('Neural Cryptography Example: ' + str(i))
    plt.legend(loc='best')
    plt.show()
