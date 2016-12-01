class Interval(object):
	''' A container object for the structure interval:
	(low, high, n_low, n_high)
	'''
	def __init__(self, low, n_low, high, n_high):
		assert n_high > n_low, "There is no eigenvalue in (%.4f, %.4f)" % (low, high)
		self.low = low
		self.n_low = n_low
		self.high = high
		self.n_high = n_high
		self.working = False

	def __contains__(self, x):
		if abs(x - self.low) < 1e-9 or abs(x - self.high) < 1e-9:
			return True
		return x >= self.low and x <= self.high

	def num_evals(self):
		return self.n_high - self.n_low

	def split(self, mid, n_mid):
		# Split the interval into potentially 2:
		# (low, mu, n_mid), (mu, high, n_mid)
		result = []
		if n_mid > self.n_low:
			result.append(Interval(self.low, self.n_low, mid, n_mid))	
		if n_mid < self.n_high:
			result.append(Interval(mid, n_mid, self.high, self.n_high))
		return result

def find_interval(mu, eintervals):
    for i in range(len(eintervals)):
        if mu in eintervals[i]:
            return eintervals.pop(i)
    return None
		