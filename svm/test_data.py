import numpy

class TestData:
    inputs = None
    targets = None
    classA = None
    classB = None
    def __init__(self, samples, seed, values=None):
        self.samples = samples
        self.seed = seed
        self.values = values

    def generate_data(self):
        # Same test data every time
        if(self.seed):
            numpy.random.seed(100)

        # Return a sample (or samples) from the “standard normal” distribution with zero mean and unit variance
        # By multiplying with a number and adding a vector we can shift this cluster to any position
        self.classA = numpy.concatenate(
        (numpy.random.randn(int(self.samples/4), 2) * 0.2 + [1.5, 0.5],
        numpy.random.randn(int(self.samples/4), 2) * 0.2 + [-1.5, 0.5]))

        # Return a sample (or samples) from the “standard normal” distribution with zero mean and unit variance
        self.classB = numpy.random.randn(int(self.samples/2), 2) * 0.2 + [0.0, -0.5]

        # Concatenate A and B
        inputs = numpy.concatenate((self.classA , self.classB))

        # N*1 array of targets, create positive and negative ones based on number of rows in A and B
        targets = numpy.concatenate(
        (numpy.ones(self.classA.shape[0]),
        -numpy.ones(self.classB.shape[0])))

        N = inputs.shape[0] # Number of rows (samples)
        # Create a list from 0-N
        permute = list(range(N))
        numpy.random.shuffle(permute)

        # Shuffle around the inputs and use slice to include both of the points for every sample
        self.inputs = inputs[permute, :]
        # Shuffle around targets
        self.targets = targets[permute]
