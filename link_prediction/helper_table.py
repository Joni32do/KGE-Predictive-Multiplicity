# Since I'm in time trouble, here a hacky solution to get the table:
import numpy as np

elements = ["Earth", "Jupiter", "Mars", "Moon", "Curiosity Rover", "Hubble", "Sun"]
x = np.array([1, 1, 1, 0.4, 0.3, 0.2, 0])
y = np.array([74.8, 77.4, 109.5, 29.1, 25.3, 50.4, 20.6])
z = np.array([2.8, 3.1, 3.4, 0.0, -1.9, -1.9, -2.9])

def range_voting(x):
    return np.interp(x, (x.min(), x.max()), (-1, 1))

print(range_voting(x))
print(range_voting(y))
print(range_voting(z))

res_range = range_voting(x) + range_voting(y) + range_voting(z)

for i in range(len(elements)):
    print(f"{elements[i]}: {res_range[i]:.2f}") 
