import numpy as np
import matplotlib.pyplot as plt

# Define the values of the scale parameter a
eta = np.linspace(0, np.pi, 10000)

# Define the values of the parameters w and rho_0/a_0^3
C= 3000
a1 =    (np.sin(eta))
a2 =   eta
a3 = np.sinh(eta)

H0 = 70
t0 = 7
t = np.linspace(0, 10, 1000)
a = np.exp((t-t0))

# Define a function to calculate the energy density as a function of a for a given value of w
pi = np.pi

# Plot the different energy density curves
plt.plot(t, a, label="w = -1")
# plt.plot(eta, a2, label="k=0")
# plt.plot(eta, a3, label="k=-1")

# Remove the tick labels from the x and y axes
plt.tick_params(axis='both', labelsize=0)

# Add axis labels and a legend
plt.xlabel("$t$ ")
plt.ylabel("$a(t)$")
# plt.xticks(np.arange(0, pi+1, step=(pi/2)), ['0','π/2', 'π'])
plt.ylim(0,3)
plt.legend()

# Show the plot
plt.show()
