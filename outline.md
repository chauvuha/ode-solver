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

Prior research has explored various methods for solving ODEs using neural networks. Some studies introduced PINNs to solve first- and second-order ODEs, highlighting their usefulness in physics simulations and their ability to incorporate physical laws directly into the model’s structure [1]. Other studies expanded on this method by modifying the loss function to include the differential equation itself, allowing the model to learn to satisfy the equation rather than simply fit example data points [2]. MathWorks presented a different strategy, using neural networks to produce closed-form approximations of ODE solutions, supported by a training process that involves generating data, defining the network, and customizing the loss function [3]. Additionally, researchers have applied similar techniques to Partial Differential Equations (PDEs), training models on randomly sampled space and time points to approximate solutions where no analytical answers exist [4]. Together, these works show the versatility and potential of neural networks in solving both ODEs and PDEs, laying the groundwork for our own project.

<br>

**References**  
[1] Amini, S., Hashemi, A., Azizi, A., & Ebrahimi, H. (2023). *Solving differential equations with Deep Learning: A beginner’s guide*. [arXiv:2302.12260](https://arxiv.org/abs/2302.12260)  
[2] Zang, Y., Bao, G., Ye, X., & Zhou, H. (2020). *Weak adversarial networks for high-dimensional partial differential equations*. [Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231220301909)  
[3] MathWorks. *Solve ODEs Using a Neural Network*. [Link](https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html)  
[4] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. [Journal of Computational Physics](https://www.sciencedirect.com/science/article/pii/S0021999118305527)

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
   - This version was less accurate due to:
     + Using a finite-difference stencil:  
       \( dNN ≈ \frac{g(x+ε) − g(x)}{ε} \)  
       which is unstable and sensitive to \( ε \).
     + Not using `tf.keras.Sequential`, resulting in less optimized training.
     + A limited loss function that only supports equations involving \( x \).

<img src="manual_pinn.png"  width="60%" />

2. **Keras PINNs:**
   - We used `Sequential` models in Keras.  
   The results for periodic functions (e.g., sine wave) were accurate.  
   However, it struggles with non-periodic ODEs due to normalization issues—activation functions may explode or vanish.

  <p align="center">
    <img src="keras_loss.png" alt="Keras Loss Curve" width="45%" />
    <img src="keras_train.png" alt="Keras Prediction vs Ground Truth" width="45%" />
  </p>

3. **DeepXDE PINNs:**
   <b>DeepXDE output for Sin(2πt):</b>
  <p align="center">
    <img src="deepxde_loss.jpg" alt="DeepXDE Loss Curve" width="45%" />
    <img src="deepxde_train.jpg" alt="Deep XDE Prediction vs Ground Truth" width="45%" />
  </p>

  - These results are excellent. The model learned:  
  \( \frac{dy}{dt} = \sin(2\pi t), \quad y(0) = 1 \)  
  almost perfectly.

  - DeepXDE simplifies defining the ODE, domain, and loss using TensorFlow’s auto-differentiation.  
  It even allows immediate error checks with exact solutions and auto-generates plots using `dde.saveplot()`.

  - By contrast, Keras or hand-built models require writing custom loss functions and sampling routines manually. DeepXDE removes boilerplate code.

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

- [Amini et al., 2023 - Solving differential equations with Deep Learning (arXiv)](https://arxiv.org/abs/2302.12260)  
- [Zang et al., 2020 - Weak adversarial networks for PDEs (Neurocomputing)](https://www.sciencedirect.com/science/article/abs/pii/S0925231220301909)  
- [MathWorks – Solve ODEs using a Neural Network](https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html)  
- [Raissi et al., 2019 – PINNs: A deep learning framework (JCP)](https://www.sciencedirect.com/science/article/pii/S0021999118305527)  
- [DeepXDE Documentation](https://deepxde.readthedocs.io/en/latest/)  
- [Keras Model API](https://keras.io/api/models/model/)  
- [Hand‑built PINN Colab Example](https://colab.research.google.com/drive/12ztGwxR1TK8Ka6H3bOsSt57kB71ieQ-W?usp=sharing)  
- [Keras & DeepXDE Comparison Colab](https://colab.research.google.com/drive/1L1EmfOFFnoXfCF8YwNxkCflEV0gAScqx?usp=sharing)

