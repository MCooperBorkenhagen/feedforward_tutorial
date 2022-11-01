import numpy
def load(filename,dtype=numpy.dtype('float'),simplify=False):
    def dropUnusedUnits(x):
        z = numpy.any(x,axis=0)
        return x[:,z]

    print(filename)
    with open(filename) as f:
        ex = numpy.array([x.strip('\n').split(',') for x in f.readlines()],dtype=dtype)

    if simplify:
        ex = dropUnusedUnits(ex)

    return ex
