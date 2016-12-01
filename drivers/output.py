import interval
import matplotlib.pyplot as plt

def display_interval(interval):
	print("(%.8f, %.8f) with %d evals." % (interval.low, interval.high, interval.num_evals()))

def display_intervals(eintervals):
	print("Current intervals information:")
	for interval in eintervals:
		display_interval(interval)
	print('\n')

def plot_interval(interval, working=False):
    color = 'green' if working else 'red'
    plt.hlines(0.0, interval.low, interval.high, color, lw=4)
    plt.vlines(interval.low, 0.03, -0.03, color, lw=2)
    plt.vlines(interval.high, 0.03, -0.03, color, lw=2)

def plot_intervals(eintervals, mu):
    fig = plt.figure(figsize=(6,6))
    plt.ylim([-1, 1])
    for interval in eintervals:
        if mu in interval:
            plot_interval(interval, True)
        else:
            plot_interval(interval)
    plt.plot([mu], [0], '*', color='black')
    plt.show()