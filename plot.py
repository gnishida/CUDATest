import matplotlib.pyplot as plt

for i in xrange(1):
	f = open("random_" + str(i) + ".txt", "r")
	list_x = []
	for line in f:
		x = line.rstrip()
		list_x.append(int(x))

	plt.hist(list_x, bins=200)
	
plt.show()