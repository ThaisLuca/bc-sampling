
import math
import numpy
import random

class Sampling:

    def __init__(self,seed=123):
        self.seed=seed

    def random(self, data, rate=0.01):
        indexes = list(range(len(data)))
        data = numpy.array(data)
        training_idx = random.sample(indexes, math.ceil(len(data)*rate))
        #test_index = list(set(indexes) - set(training_idx))
        return data[training_idx]