import scipy.stats as stats
from scipy import sqrt, exp, pi


def ownNormalPdf(x, mean=0.0, sigma=1.0):
    return 1 / sqrt(2 * pi * sigma ** 2) * exp(-(x - mean) ** 2 / sigma ** 2 / 2)  # verify manually


import scipy.stats as stats
from scipy import sqrt, exp, pi

d1 = stats.norm.pdf(0, 0.1, 0.05)
print("d1=", d1)

d2 = 1 / sqrt(2 * pi * 0.05 ** 2) * exp(-(0.1) ** 2 / 0.05 ** 2 / 2)  # verify manually
print("d2=", d2)

d3 = ownNormalPdf(x=0, mean=0.1, sigma=0.05)

# graph
import scipy as sp
import matplotlib.pyplot as plt

x = sp.arange(-3, 3, 0.1)
y = sp.stats.norm.pdf(x)
plt.title("Standard Normal Distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y)
plt.show()


# graph with annotation for some z value
import scipy as sp
import matplotlib.pyplot as plt

x = sp.arange(-3, 3, 0.1)
y = ownNormalPdf(x)  # or y = sp.stats.norm.pdf(x)

z = -2.325
xStart = -3.8
yStart = .2
xEnd = -2.5
yEnd = .05

plt.ylim(0, 0.45)
plt.plot(x, y)
x2 = sp.arange(-4, z, 1 / 40.)
sum = 0
delta = 0.05

s = sp.arange(-10, z, delta)
for i in s:
    sum += ownNormalPdf(i) * delta
plt.annotate('area is ' + str(round(sum, 4)), xy=(xEnd, yEnd), xytext=(xStart, yStart),
             arrowprops=dict(facecolor='red', shrink=0.01))
plt.annotate('z= ' + str(z), xy=(z, 0.01))
plt.fill_between(x2, ownNormalPdf(x2))
plt.show()
