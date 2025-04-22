<script type="text/javascript"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Differential Equation Solver

## Members  
- Eve Zeng  
- Chau Vu  
- Dualeh<sup>2</sup>  
- Sultan Aldawodi 



## Project Outline

### Abstract
We will summarize the motivation, methods, and results of our project. (To Be Written Later)

### Introduction
We will introduce Ordinary Differential Equations (ODEs) and explain their importance in modeling real-world systems. We will motivate the use of neural networks, especially Physics-Informed Neural Networks (PINNs), as a way to solve ODEs and improve interpretability.

### Ethics Discussion
We will address the ethical considerations of releasing a public-facing solver, including risks of academic dishonesty. We will also explain our mitigation strategies, such as promoting transparency and step-by-step breakdowns, to emphasize the tool’s use as a learning aid rather than a shortcut.

### Related Works
We will review prior research on neural network-based differential equation solvers. This will include papers on PINNs, loss function techniques, analytic approximations, and PDE solvers. We will compare these to our own project, which combines PINNs with pattern matching and aims for public accessibility and interpretability.

### Methods

The primary software we use to implement the PINN is [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/). The neural network is built using the Dense and Input layers from Keras, with the Adam optimizer used to minimize the loss function. We will be training using Jupyter Notebook or Google Colab. 

For the dataset, we constructed training data by sampling from various ODEs. For example, for a first-order ODE, such as:

$$
\frac{dy}{dx} + y = 0
$$

The exact solution for it, like:

$$
y(x) = e^{-x}
$$

was used for comparison.

The dataset consists of 100–2000 data points sampled from a defined domain range (e.g., [0, 5] or [-2, 2]). The model is trained to minimize the difference between the predicted solution and the exact solution by updating the network’s parameters using backpropagation.

Some of the more challenging parts of this project will be trying to understand the structure of the neural network and implementing it to our use. For PINNs, the loss function separately considers both the differential equation solution value and its initial condition value relative to the expected. 

Our loss function is thus complicated and could be hard to debug. If all goes well, we could explore the possibilities of modifying the loss function or the other training procedures to compare the results. 



### Discussion
We are creating our own data set, with methods provided by <a href = "https://github.com/rtqichen/torchdiffeq/blob/master/README.md"><i>torchdiffeq</i></a>, and trained our PINN with these specifically generated data set. 
We are implementing this PINN network to train three different data sets, corresponding to three different types of differential equations, based on <a href = "https://github.com/rtqichen/torchdiffeq">these tests</a>. 
We will also create a graph visualization to show how well our neural network’s predictions align with the ground truth solutions of the differential equation during the training process.
After training these Neural Networks, we will again generate another set of data by similar methods, and test each of these three networks on their accuracy. 
We will compare our base type differential equation to the literature results, expecting to perform less accurately due to less data. We will also compare the accuracy between each type of NN, and decipher the potential reasons that one does better or worse. 
In the future, we would spend more time to figure out how to generalize our neural network to more types of equations.

### Results
1. **PINNs built by hand (Non-Keras or XDE):**
   - We found that this version is significantly less accurate than others because:
     + In our loss function, instead of using `tf.GradientTape(u, t)` like the Keras version, we use a finite‐difference stencil \( dNN \approx \frac{g(x + \epsilon) - g(x)}{\epsilon} \), which is slower, less stable, and inherently unreliable. Accuracy critically depends on choosing an optimal \( \epsilon \); if it’s too large, you miss details, and if it’s too small, floating-point noise dominates.
     + We’re not using `tf.keras.Sequential`, a model that has been developed and optimized for these tasks. Instead, we manually define our weights/biases and `tf.matmul` calls, which might be slower and more error-prone during training.

   - For this model, our loss function is designed to use a predefined \( f(x) \) function, which limits the model to solving equations that involve only \( x \). As a result, the model is less flexible because it cannot handle ODEs that include both \( x \) and \( y \) or other variable interactions.

[Add the graph here]

2. **Keras PINNs:**

3. **DeepXDE PINNs:**




### Conclusion/Future Work
We will summarize our contributions and findings, reiterating our focus on interpretability and accessibility. We will also outline future directions, such as expanding to PDEs, refining the interface, or improving model generalizability.

### References
We will list all cited works using markdown footnotes, properly formatted. This includes the related papers on PINNs, PDEs, MathWorks implementations, and other neural network approaches.



## Introduction

ODEs (ordinary differential equations) are an important way of modeling the world, showing up in different fields such as economics, biology, and physics. Being able to have accurate solutions to these equations gives power in analyzing complex systems. Our project focuses on testing a branch of neural network formalism called PINN (Physics informed neural networks) in its ability to solve differential equations, such that these equations have more interpretable solutions. This will allow more information to be gained from the solutions, and will also be clear and helpful to those who are looking for patterns and structures in these equations. We plan to develop a program based on previous research findings and target it towards a more pedagogical approach, hoping to help students understand ODE methods. Our final product will consist of a website hosting the trained and implemented Neural Network, descriptions, documentations, and methods on the theories behind our program and the mathematics. 


## Related Work

Prior research has explored various methods for solving ODEs using neural networks. Some studies introduced PINNs to solve first- and second-order ODEs, highlighting their usefulness in physics simulations and their ability to incorporate physical laws directly into the model’s structure[^1]. Other studies expanded on this method by modifying the loss function to include the differential equation itself, allowing the model to learn to satisfy the equation rather than simply fit example data points[^2]. MathWorks presented a different strategy, using neural networks to produce closed-form approximations of ODE solutions, supported by a training process that involves generating data, defining the network, and customizing the loss function[^3]. Additionally, researchers have applied similar techniques to Partial Differential Equations (PDEs), training models on randomly sampled space and time points to approximate solutions where no analytical answers exist[^4]. Together, these works show the versatility and potential of neural networks in solving both ODEs and PDEs, laying the groundwork for our own project.


## Project Design

Our project distinguishes itself from existing research in neural network-based differential equation solvers through its focus on both pattern matching and Physics-Informed Neural Network (PINN) methodologies. While prior work has explored neural networks for solving ODEs, much of it has concentrated on either direct function approximation or strictly physics-informed approaches. Our approach integrates both perspectives, allowing us to evaluate their respective strengths and limitations in solving ODEs. Additionally, rather than solely focusing on performance benchmarks, we aim to compare how neural network-generated solutions align with traditional human-solving strategies, providing a deeper interpretability component. That way users can both get a solution to a differential equation and see the steps taken to solve it. That way, even if the solution is incorrect, the user can still find where the neural network went wrong through visual compute graphs.

 
## Ethical & Educational Considerations

Furthermore, our project seeks to go beyond academic research by making our findings accessible through an interactive web interface, ensuring that anyone can use the program upon publishing our results. We understand there is the risk of students misusing this program in ways that violate school policies, such as cheating and plagiarism, but the applicability of this program in helping students learn and understand differential equations outweighs the chances of misuse, as it provides a learning opportunity for people who may not have access to advanced calculators.


## Conclusion

In conclusion, our project expands on existing research by integrating pattern matching and PINN methodologies while prioritizing interpretability and accessibility. By providing visual compute graphs and a free web-based interface, we make neural network-based ODE solving both transparent and widely available. While misuse is a concern, the educational benefits like helping users understand and verify solutions outweigh the risks, making this a valuable tool for students and researchers alike.


## References

[^1]: Amini, S., Hashemi, A., Azizi, A., & Ebrahimi, H. (2023). *Solving differential equations with Deep Learning: A beginner’s guide*. arXiv preprint arXiv:2302.12260. [View paper](https://arxiv.org/abs/2302.12260)

[^2]: Zang, Y., Bao, G., Ye, X., & Zhou, H. (2020). *Weak adversarial networks for high-dimensional partial differential equations*. Neurocomputing, 399, 305–315. [View paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220301909)

[^3]: MathWorks. (n.d.). *Solve ODEs Using a Neural Network*. [View paper](https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html)

[^4]: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. *Journal of Computational Physics*, 378, 686–707. [View paper](https://www.sciencedirect.com/science/article/pii/S0021999118305527)

