import random


# equal sampling of categories in train and test sets
def sample_equally(d, t):
    pos_inds = [i for i in range(len(t)) if t[i] == 1]
    zero_inds = [i for i in range(len(t)) if t[i] == 0]
    num_pos = len(pos_inds)
    num_neg = len(zero_inds)
    if num_pos > num_neg:
        pos_inds = random.sample(pos_inds, num_neg)
    else:
        zero_inds = random.sample(zero_inds, num_pos)
    d_tmp = [d[i] for i in pos_inds]
    d = d_tmp + [d[i] for i in zero_inds]
    t_tmp = [t[i] for i in pos_inds]
    t = t_tmp + [t[i] for i in zero_inds]
    return d, t
