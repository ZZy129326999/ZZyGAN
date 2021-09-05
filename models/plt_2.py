from pandas import Series,DataFrame
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
df = DataFrame([
    [95, 88, 90],
    [99, 94, 92],
    [99, 94, 97],
    [95, 92, 94],
    [99, 92, 96]
],columns=['Validation accuracy(a)', 'Validation accuracy(b)','Validation accuracy(c)'],
               index = ['AlexNet', 'DenseNet-121', 'ResNet-50', 'VGG-16', 'X-ception'])
df.plot(kind='bar', rot=0)
# plt.show()
plt.savefig('./validimg.png')