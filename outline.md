<script type="text/javascript"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<style>
  details summary {
    cursor: pointer;
    font-weight: 600;
    color: #2c3e50;
    transition: color 0.2s;
    font-size: 1.25rem;
    margin-bottom: 0.5rem;
  }
  details summary:hover {
    color: #1abc9c;
  }
  details summary::-webkit-details-marker {
    font-size: 1.4em;
  }
  details summary::marker {
    font-size: 1.4em;
  }
  details {
    margin-bottom: 1.5rem;
  }
</style>

# Differential Equation Solver

**Members**  
Eve Zeng · Chau Vu · Dualeh² · Sultan Aldawodi

<details>
<summary>Abstract</summary>

Physics‐informed neural networks (PINNs) offer a mesh‐free, data‐efficient approach to obtaining approximate solutions of ordinary differential equations (ODEs) by embedding the governing equations directly into the loss function of a neural network. In this project, we compare three distinct PINN implementations for solving benchmark ODEs: (i) a from‐scratch fully connected network coded in plain Python, (ii) a Keras‐based PINN leveraging TensorFlow’s high‐level APIs, and (iii) a DeepXDE model utilizing its specialized automatic differentiation and domain-decomposition features. These models were trained to solve simple ODEs, and their results were compared. The Keras and DeepXDE models performed with high accuracy, though the construction of the network needs to be modified with initial conditions to accommodate different scenarios. Still, these networks and their results demonstrate the reliable use of PINNs to solve ordinary differential equations and have a promising future in tackling complex partial differential equations with no analytical solutions.

</details>

<details>
<summary>Introduction</summary>

ODEs (ordinary differential equations) are an important way of modeling the world in different fields such as economics, biology, and physics. For example, the physics of fluid dynamics is governed by the Navier–Stokes partial differential equations. Having accurate solutions to these equations gives power in analyzing complex systems. Our project focuses on testing a branch of neural network formalism called PINN (Physics‐informed neural networks) in its ability to solve differential equations, such that these equations produce simple and accurate results for difficult problems. Some of the hardest differential equations cannot be analytically solved, so having a reliable approximate solution from a neural network can help build complex models and systems. Our project tests three different kinds of PINNs, starting with a simple, made-from-scratch model using a fully connected neural network, then a Keras PINN via TensorFlow, and finally a DeepXDE model. Our tests on simple ODEs show promising signs for tackling more complex PDEs in future work.

</details>

<details>
<summary>Ethics Discussion</summary>

Our project seeks to go beyond academic research by making our findings accessible through an interactive web interface, ensuring that anyone can use the program upon publishing our results. We understand there is the risk of students misusing this program in ways that violate school policies, such as cheating and plagiarism, but the applicability of this program in helping students learn and understand differential equations outweighs the chances of misuse, as it provides a learning opportunity for people who may not have access to advanced calculators.

</details>

<details>
<summary>Related Work</summary>

Prior research has explored various methods for solving ODEs using neural networks. Some studies introduced PINNs to solve first- and second-order ODEs, highlighting their usefulness in physics simulations and their ability to incorporate physical laws directly into the model’s structure <a href="#ref1">[1]</a>. Other studies expanded on this method by modifying the loss function to include the differential equation itself <a href="#ref2">[2]</a>. MathWorks presented a different strategy, using neural networks to produce closed-form approximations of ODE solutions <a href="#ref3">[3]</a>. Additionally, researchers have applied similar techniques to PDEs, training models on randomly sampled space and time points to approximate solutions where no analytical answers exist <a href="#ref4">[4]</a>. These works show the versatility and potential of neural networks in solving both ODEs and PDEs, laying the groundwork for our own project.

</details>



<details>
<summary>Methods</summary>

The primary software we use to implement the PINN is TensorFlow and Keras. We will train three PINNs: a manually-built neural network, a Keras-based PINN using automatic differentiation, and a DeepXDE library that automates the setup and training of the neural network. The hand-built network is built using the Dense and Input layers from Keras, with the Adam optimizer used to minimize the loss function, which combines the residual of the differential equation with the error from the initial or boundary conditions.

For the dataset, we constructed training data by sampling from various ODEs. For example, for the first-order ODE:  
\[
  \frac{dy}{dx} + y = 0,
\]  
The exact solution is:  
\[
  y(x) = e^{-x}
\]

</details>

<details>
<summary><strong>Discussion</strong></summary>

We are creating our own data set using methods from [torchdiffeq](https://github.com/rtqichen/torchdiffeq), and trained our PINN with these specifically generated data sets.  
We implemented this PINN network to train three different types of differential equations, based on [these tests](https://github.com/rtqichen/torchdiffeq).  
We will create graphs to visualize how well our neural network’s predictions align with the ground truth during training.  
After training, we’ll generate new data and test each model's accuracy.  
We expect our hand-built model to perform worse due to limited data and less optimization, but it will give insight into the tradeoffs.  
We also compare the accuracy of all three NNs and analyze why one may perform better than the others.  
In the future, we aim to generalize these networks to solve more diverse equations.

</details>

### Results
1. **PINNs built by hand (Non-Keras or XDE):**
   - We found that this version is significantly less accurate than others because:
     + In our loss function, instead of using `tf.GradientTape(u, t)` like the Keras version, we use a finite‐difference stencil dNN ≈ (g(x+ε) − g(x)) / ε, which is slower, less stable, and inherently unreliable. Accuracy critically depends on choosing an optimal \( \epsilon \); if it’s too large, you miss details, and if it’s too small, floating-point noise dominates.
     + We’re not using `tf.keras.Sequential`, a model that has been developed and optimized for these tasks. Instead, we manually define our weights/biases and `tf.matmul` calls, which might be slower and more error-prone during training.

   - For this model, our loss function is designed to use a predefined \( f(x) \) function, which limits the model to solving equations that involve only \( x \). As a result, the model is less flexible because it cannot handle ODEs that include both \( x \) and \( y \) or other variable interactions.

<img src="manual_pinn.png"  width="60%" />

2. **Keras PINNs:**
  - The Keras package has existing functions that provides existing and established NN models. We used the sequential model provided by the Keras package. 
  This model did well with our given example of a Sine wave, the original function and the NN approximation matched each other almost perfectly. 
  However, problems arise when we try other equations that are not periodic. It is hard to normalize the equation when it goes to infinity, but not normalizing it could risk other problems to the activation function blowing up or going to zero. Thus, this is a problem that needs to be addressed for a better model of training all kinds of differential equation NN, not just the periodic ones. 

  <p align="center">
    <img src="keras_loss.png" alt="Keras Loss Curve" width="45%" />
    <img src="keras_train.png" alt="Keras Prediction vs Ground Truth" width="45%" />
  </p>

3. **DeepXDE PINNs:**
  <b>DeepXDE output for Sin(2*pi*t):</b>
  <p align="center">
    <img src="deepxde_loss.jpg" alt="DeepXDE Loss Curve" width="45%" />
    <img src="deepxde_train.jpg" alt="Deep XDE Prediction vs Ground Truth" width="45%" />
  </p>

  - These results look spot on. The loss curves drop smoothly for both train and test, and the solution plot shows that the PINN learned:  
\( \frac{dy}{dt} = \sin(2\pi t), \quad y(0) = 1 \)  
almost perfectly. The red dashed line overlaps the true black curve and the training dots.

  - Additionally, DeepXDE makes this entire process almost trivial. We can define the ODE, domain, and initial condition in a few lines, and DeepXDE uses TensorFlow’s automatic differentiation to build a loss that enforces the differential equation and boundary conditions. It even lets us plug in an exact solution for immediate error checks. A single call to `dde.saveplot()` then gives us both the loss history and the prediction‑vs‑truth comparison without any extra plotting code.

  - By contrast, if we tried the same thing in Keras or with a hand‑rolled feedforward network, we’d have to write custom loss functions, call gradient routines ourselves, manually sample time points, and wire up all the training loops. DeepXDE abstracts all that away, so we can focus on modeling rather than boilerplate—and for anyone solving ODEs or PDEs, that makes it the fastest, most reliable choice.
  
<details>
<summary><strong>DeepXDE Code</strong></summary>

```python
from deepxde.backend.set_default_backend import set_default_backend
set_default_backend("tensorflow")
import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import math as m
pi = tf.constant(m.pi)

def ode_system(t, u):
   du_t = dde.grad.jacobian(u, t)
   return du_t - tf.math.sin(2*pi*t)

def boundary(t, on_initial):
   return on_initial and np.isclose(t[0], 0)

geom = dde.geometry.TimeDomain(0, 2)
ic = dde.IC(geom, lambda t: 1, boundary)

def true_solution(t):
   return -np.cos(2 * np.pi * t) / (2 * np.pi) + (1 + 1 / (2 * np.pi))

data = dde.data.PDE(geom,
                   ode_system,
                   ic,
                   num_domain=30,
                   num_boundary=2,
                   solution=true_solution,
                   num_test=100)

layer_size = [1, 32, 32, 1]
activation = "tanh"
initializer = "Glorot uniform"

NN = dde.maps.FNN(layer_size, activation, initializer)
model = dde.Model(data, NN)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=3000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)
</details>
<details> <summary><strong>Keras Code</strong></summary>
python
Copy
Edit
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import warnings
warnings.filterwarnings("ignore")

NN = tf.keras.models.Sequential([
   tf.keras.layers.Input((1,)),
   tf.keras.layers.Dense(32, activation='tanh'),
   tf.keras.layers.Dense(32, activation='tanh'),
   tf.keras.layers.Dense(1)
])

NN.summary()
optm = tf.keras.optimizers.Adam(learning_rate=0.001)

def ode_system(t, net):
   t = t.reshape(-1,1)
   t = tf.constant(t, dtype=tf.float32)
   t_0 = tf.zeros((1,1))
   one = tf.ones((1,1))

   with tf.GradientTape() as tape:
       tape.watch(t)
       u = net(t)
       u_t = tape.gradient(u, t)

   ode_loss = u_t - tf.math.sin(2*np.pi*t)
   IC_loss = net(t_0) - one
   square_loss = tf.square(ode_loss) + tf.square(IC_loss)
   total_loss = tf.reduce_mean(square_loss)
   return total_loss

train_loss_record = []

for itr in range(3000):
   train_t = (np.random.rand(20)*2).reshape(-1, 1)
   with tf.GradientTape() as tape:
       train_loss = ode_system(train_t, NN)
       train_loss_record.append(train_loss)
       grad_w = tape.gradient(train_loss, NN.trainable_variables)
       optm.apply_gradients(zip(grad_w, NN.trainable_variables))
   if itr % 1000 == 0:
       print('Epoch: {} Loss: {:.4f}'.format(itr, train_loss.numpy()))

plt.figure(figsize=(6, 4))
plt.plot(train_loss_record)
plt.show()

test_t = np.linspace(0, 2, 100)
test_t = tf.constant(test_t, dtype=tf.float32)

pred_u = NN(test_t).numpy()
true_u = -np.cos(2*np.pi*test_t)/(2*np.pi) + (1+1/(2*np.pi))

plt.figure(figsize=(6, 4))
plt.plot(test_t, true_u, 'k', label='True', alpha=0.3)
plt.plot(test_t, pred_u, '--r', label='Prediction', linewidth=3)
plt.legend()
plt.xlabel('t')
plt.ylabel('u')
plt.show()
</details>
<details> <summary><strong>PINNs Built by Hand (Non-Keras or XDE)</strong></summary>
python
Copy
Edit
# Initial condition
f0 = 1
inf_s = np.sqrt(np.finfo(np.float32).eps)

# Parameters
learning_rate = 0.01
training_steps = 500
batch_size = 100
display_step = training_steps / 10

# Network parameters
n_input, n_hidden_1, n_hidden_2, n_output = 1, 32, 32, 1

weights = {
   'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
   'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
   'out': tf.Variable(tf.random.normal([n_hidden_2, n_output]))
}
biases = {
   'b1': tf.Variable(tf.random.normal([n_hidden_1])),
   'b2': tf.Variable(tf.random.normal([n_hidden_2])),
   'out': tf.Variable(tf.random.normal([n_output]))
}

optimizer = tf.optimizers.SGD(learning_rate)

def multilayer_perceptron(x):
   x = np.array([[[x]]], dtype='float32')
   layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
   layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
   output = tf.matmul(layer_2, weights['out']) + biases['out']
   return output

def g(x):
   return x * multilayer_perceptron(x) + f0

pi = tf.constant(np.pi)
def f(x):
   return tf.math.sin(2*pi*x)

def custom_loss():
   xs = np.random.rand(20)
   errors = []
   for x in xs:
       dNN = (g(x + inf_s) - g(x)) / inf_s
       errors.append((dNN - f(x))**2)
   return tf.reduce_sum(errors)

def train_step():
   with tf.GradientTape() as tape:
       loss = custom_loss()
   trainable_variables = list(weights.values()) + list(biases.values())
   gradients = tape.gradient(loss, trainable_variables)
   optimizer.apply_gradients(zip(gradients, trainable_variables))
</details> ```

<details>
<summary>Conclusion/Future Work</summary>

This project opens up opportunities for future work. We aim to create better models that handle higher-order and complex systems.  
One idea is to combine the strengths of different models into a meta-model that selects the best PINN for a given equation.  
We also plan to build a web-based demo to showcase the models and compare performance.

</details>

<details>
<summary>Reflection</summary>

Our team learned not only how to implement PINNs but also how to research, collaborate, and plan development work.  
We improved our literature review skills, modified open-source models, and compared performance across different approaches.  
This project also helped us learn to work as a group and prioritize tasks.

</details>

---

## References


<ol>
  <li id="ref1">
    Amini, S., Hashemi, A., Azizi, A., & Ebrahimi, H. (2023). <i>Solving differential equations with Deep Learning: A beginner’s guide</i>. <a href="https://arxiv.org/abs/2302.12260" target="_blank">arXiv</a>.
  </li>
  <li id="ref2">
    Zang, Y., Bao, G., Ye, X., & Zhou, H. (2020). <i>Weak adversarial networks for high-dimensional partial differential equations</i>. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231220301909" target="_blank">Neurocomputing</a>.
  </li>
  <li id="ref3">
    MathWorks. <i>Solve ODEs Using a Neural Network</i>. <a href="https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html" target="_blank">Documentation</a>.
  </li>
  <li id="ref4">
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). <i>Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations</i>. <a href="https://www.sciencedirect.com/science/article/pii/S0021999118305527" target="_blank">Journal of Computational Physics</a>.
  </li>
</ol> 
- [DeepXDE Documentation](https://deepxde.readthedocs.io/en/latest/)  
- [Keras Model API](https://keras.io/api/models/model/)  
- [Hand‑built PINN Colab Example](https://colab.research.google.com/drive/12ztGwxR1TK8Ka6H3bOsSt57kB71ieQ-W?usp=sharing)  
- [Keras & DeepXDE Comparison Colab](https://colab.research.google.com/drive/1L1EmfOFFnoXfCF8YwNxkCflEV0gAScqx?usp=sharing)

