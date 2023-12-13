import numpy as np
import matplotlib.pyplot as plt
# Define the activation functions as per the second image provided
def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def gcu(z):
    return z * np.cos(z)

def squ(z):
    return z**2 + z

def prelu(z, alpha=0.25):
    return np.maximum(alpha*z, z)

def elu(z, alpha=1):
    return np.where(z < 0, alpha*(np.exp(z) - 1), z)

# Create a range of x values
x_values = np.linspace(-3, 3, 300)

# Compute the activation functions using the equations from the second picture
y_relu = relu(x_values)
y_tanh = tanh(x_values)
y_gcu = gcu(x_values)
y_squ = squ(x_values)
y_prelu = prelu(x_values)
y_elu = elu(x_values, alpha=1)  # Alpha is taken from the ELU function provided in the second picture

# Plot the activation functions
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_relu, label='RELU')
plt.plot(x_values, y_tanh, label='Tanh')
plt.plot(x_values, y_gcu, label='GCU')
plt.plot(x_values, y_squ, label='SQU')
plt.plot(x_values, y_prelu, label='PRELU', linestyle='--')  # Dashed line for visibility
plt.plot(x_values, y_elu, label='ELU', linestyle='-.')  # Dash-dot line for visibility

# Add labels and legend
plt.xlabel('x')
plt.ylabel('f(z)')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()