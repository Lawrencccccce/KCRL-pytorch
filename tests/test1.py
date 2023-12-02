import numpy as np


maxlen = 10
baseint = 2**maxlen
graph_to_int = []
graph_to_int2 = []
graph_batch = np.random.randint(2, size=(maxlen, maxlen))
print(graph_batch)

for i in range(maxlen):
            graph_batch[i][i] = 0
            tt = np.int32(graph_batch[i])
            graph_to_int.append(baseint * i + int(''.join([str(ad) for ad in tt]), 2))
            graph_to_int2.append(int(''.join([str(ad) for ad in tt]), 2))

print(graph_to_int)
print(graph_to_int2)