from pandas import Series,DataFrame
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
df = DataFrame([
    [92, 88, 94],
    [96, 91, 98],
    [94, 92, 96],
    [93, 90, 95],
    [93, 90, 94]
],columns=['Test accuracy(a)', 'Test accuracy(b)','Test accuracy(c)'],
               index = ['AlexNet', 'DenseNet-121', 'ResNet-50', 'VGG-16', 'X-ception'])
# df.set_index(['Region', 'Region', 'Region', 'Region', 'Region'])
df.plot(kind='bar', rot=0)
# plt.show()
plt.savefig('./testimg.png')