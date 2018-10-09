
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

my_scalar = tf.placeholder('float32')

scalar_squared = my_scalar ** 2

# A derivative of a scalar_squared by my_scalar
derivative = tf.gradients(scalar_squared, [my_scalar,])
s = tf.InteractiveSession()

x = np.linspace(-3,3)

x_squared, x_squared_der = s.run([scalar_squared, derivative[0]], {my_scalar: x})

print(x_squared, x_squared_der)


# Now plot the two lists 

plt.figure(1)
plt.plot(x, x_squared, label = "$x^2$")

plt.plot(x, x_squared_der, label = r"$\frac{dx^2}{dx}$")

plt.title("X Squared and its Derivative")
plt.legend()

plt.show()

my_vector = tf.placeholder('float32', [None])
# Compute the gradient of the next weird function over my_scalar and my_vector
# Warning! Trying to understand the meaning of that function may result in permanent brain damage
weird_psychotic_function = tf.reduce_mean(
    (my_vector+my_scalar)**(1+tf.nn.moments(my_vector,[0])[1]) + 
    1./ tf.atan(my_scalar))/(my_scalar**2 + 1) + 0.01*tf.sin(
    2*my_scalar**1.5)*(tf.reduce_sum(my_vector)* my_scalar**2
                      )*tf.exp((my_scalar-4)**2)/(
    1+tf.exp((my_scalar-4)**2))*(1.-(tf.exp(-(my_scalar-4)**2)
                                    )/(1+tf.exp(-(my_scalar-4)**2)))**2

der_by_scalar = tf.gradients(weird_psychotic_function, my_scalar)
der_by_vector = tf.gradients(weird_psychotic_function, my_vector)

# Plotting the derivative
scalar_space = np.linspace(1, 7, 100)

y = [s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 2, 3]})
     for x in scalar_space]

plt.figure(2)

plt.plot(scalar_space, y, label='function')

y_der_by_scalar = [s.run(der_by_scalar,
                         {my_scalar:x, my_vector:[1, 2, 3]})
                   for x in scalar_space]
plt.plot(scalar_space, y_der_by_scalar, label='derivative')
plt.grid()
plt.legend();
plt.show()

''' Notes
Learn:
matplotlib animation and rc
matplotlib_utils
how pyplot.subplots() works

np.meshgrid(x_var, y_var)  ---> 



'''









