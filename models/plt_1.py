from pandas import Series,DataFrame
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
df = DataFrame([
    [100, 100, 100],
    [94, 100, 100],
    [100, 100, 100],
    [100, 100, 100],
    [100, 100, 100]
],columns=['Training accuracy(a)', 'Training accuracy(b)','Training accuracy(c)'],
               index = ['AlexNet', 'DenseNet-121', 'ResNet-50', 'VGG-16', 'X-ception'])
# df.set_index(['Region', 'Region', 'Region', 'Region', 'Region'])
df.plot(kind='bar', rot=0)
# plt.show()
plt.savefig('./trainimg.png')

