import random as rand

def create_bag(samples, n):
    '''
    @param samples, the aggregate data set
    @param n, the cardinality of the bag
    @return the bag
    '''
    bag  = rand.choices(samples, k=n)
    return bag