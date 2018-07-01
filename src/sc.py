import numpy as np
from collections import Counter


# consensus-based (democratic) strategies
def my_additive_util(ratings, rankings):
    unique = np.unique(rankings)
    sums = np.zeros(len(unique))
    for i in range(len(ratings)):
        for j in range(len(ratings[0])):
            sums[np.where(unique==rankings[i][j])] += ratings[i][j]
    return unique[np.argsort(sums)[::-1]]

def additive_util(ratings, n):
	sums = np.sum(ratings, axis=0)
	rec = np.argsort(sums)[::-1][:n]
	return rec


def avg_without_misery(ratings, n, misery):
	ind = np.where(ratings <= misery)
	items = np.unique(ind[1])
	avg = np.average(ratings, axis=0)
	rec = np.argsort(avg)[::-1]
	rec = list(rec)
	for v in items:
		rec.remove(v)
	return rec[:n]


# borderline (role based, dictatorship) strategies

def least_misery(ratings, n):
	mins = np.min(ratings, axis=0)
	rec = np.argsort(mins)[::-1][:n]
	return rec


def most_pleasure(ratings, n):
	maxes = np.max(ratings, axis=0)
	rec = np.argsort(maxes)[::-1][:n]
	return rec


# majority based strategies
def my_borda_count(rankings):
    unique = np.unique(rankings)
    sums = np.zeros(len(unique))
    ind_range = np.arange(len(unique))
    for r in rankings:
        inds = []
        for i, item in enumerate(r):
            ind = np.where(unique==item)[0][0]
            inds.append(ind)
            sums[ind] += i
        sums[np.invert(np.isin(ind_range, inds))] += len(r)
    rec = unique[np.argsort(sums)]
    return rec

def borda_count(ratings, n):
	sums = np.zeros(ratings.shape[1])
	for r in ratings:
		indices = list(range(len(r)))
		# print("r: ", r)
		indices.sort(key=lambda x: r[x])
		c = Counter(list(r))
		ranks = list(range(len(r)))
		curr_rank = 0
		seen = False
		for i, val in enumerate(indices):
			el = r[val]
			if c[el] > 1 and not seen:
				curr_rank += 0.5
				seen = True
			ranks[val] = curr_rank
			if c[el] == 1:
				curr_rank += 1
		# print(ranks)
		sums = sums + ranks
	# print(sums)
	rec = np.argsort(sums)[::-1][:n]
	return rec


def copeland_rule(ratings, n):
	l = ratings.shape[1]
	cope = np.zeros([l, l])
	for i in range(l):
		cope[i, i] = 0
		for j in range(i):
			item1 = ratings[:, j]
			item2 = ratings[:, i]
			# print(item1 > item2)
			c = Counter(item1 > item2).most_common()[0][0]
			res = 1 if c else -1
			cope[i, j] = res
			cope[j, i] = -res

	# print(cope)
	sums = np.sum(cope, axis = 1)
	# print(sums)
	recs = np.argsort(sums)[::-1]
	n = n if n > 0 else recs.shape[0]
	return recs[:n]


def main():
	# Rating rows->users, cols-> items
	rates = np.array([[10, 4, 3, 6, 9], [1, 9, 8, 9, 7], [10, 5, 2, 7, 9]])
	print(rates)
	n = 3
	# recs = []
	print(additive_util(rates, n))
	print(avg_without_misery(rates, n, 5))
	print(least_misery(rates, n))
	print(most_pleasure(rates, n))
	print(borda_count(rates, n))
	print(copeland_rule(rates, n))

#if __name__ == '__main__':
#	main()
