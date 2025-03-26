# Differential Equation Solver

## Members  
- Eve Zeng  
- Chau Vu  
- Dualeh<sup>2</sup>  
- Sultan Al Dawoodi  



## Project Outline

Abstract
We will summarize the motivation, methods, and results of our project. (To Be Written Later)

Introduction
We will introduce Ordinary Differential Equations (ODEs) and explain their importance in modeling real-world systems. We will motivate the use of neural networks, especially Physics-Informed Neural Networks (PINNs), as a way to solve ODEs and improve interpretability.

Ethics Discussion
We will address the ethical considerations of releasing a public-facing solver, including risks of academic dishonesty. We will also explain our mitigation strategies, such as promoting transparency and step-by-step breakdowns, to emphasize the tool’s use as a learning aid rather than a shortcut.

Related Works
We will review prior research on neural network-based differential equation solvers. This will include papers on PINNs, loss function techniques, analytic approximations, and PDE solvers. We will compare these to our own project, which combines PINNs with pattern matching and aims for public accessibility and interpretability.

Methods
We will describe the technical aspects of our project, including how we train the neural networks, what datasets we use, and how we evaluate results. 

Discussion/Results
We will present and analyze the performance of different neural network implementations. We will discuss how closely their solutions match traditional methods, highlight any limitations or failures, and evaluate the overall educational value of the system.

Conclusion/Future Work
We will summarize our contributions and findings, reiterating our focus on interpretability and accessibility. We will also outline future directions, such as expanding to PDEs, refining the interface, or improving model generalizability.

References
We will list all cited works using markdown footnotes, properly formatted. This includes the related papers on PINNs, PDEs, MathWorks implementations, and other neural network approaches




## Introduction

ODEs (ordinary differential equations) are an important way of modeling the world, showing up in different fields such as economics, biology, and physics. Being able to have accurate solutions to these equations gives power in analyzing complex systems. Our project focuses on testing a branch of neural network formalism called PINN (Physics informed neural networks) in its ability to solve differential equations, such that these equations have more interpretable solutions. This will allow more information to be gained from the solutions, and will also be clear and helpful to those who are looking for patterns and structures in these equations. We plan to develop a program based on previous research findings and target it towards a more pedagogical approach, hoping to help students understand ODE methods. Our final product will consist of a website hosting the trained and implemented Neural Network, descriptions, documentations, and methods on the theories behind our program and the mathematics. 


## Related Work

Prior research has explored various methods for solving ODEs using neural networks. Some studies introduced PINNs to solve first- and second-order ODEs, highlighting their usefulness in physics simulations and their ability to incorporate physical laws directly into the model’s structure (1). Other studies expanded on this method by modifying the loss function to include the differential equation itself, allowing the model to learn to satisfy the equation rather than simply fit example data points (2). MathWorks presented a different strategy, using neural networks to produce closed-form approximations of ODE solutions, supported by a training process that involves generating data, defining the network, and customizing the loss function (3). Additionally, researchers have applied similar techniques to Partial Differential Equations (PDEs), training models on randomly sampled space and time points to approximate solutions where no analytical answers exist (4). Together, these works show the versatility and potential of neural networks in solving both ODEs and PDEs, laying the groundwork for our own project. 


## Project Design

Our project distinguishes itself from existing research in neural network-based differential equation solvers through its focus on both pattern matching and Physics-Informed Neural Network (PINN) methodologies. While prior work has explored neural networks for solving ODEs, much of it has concentrated on either direct function approximation or strictly physics-informed approaches. Our approach integrates both perspectives, allowing us to evaluate their respective strengths and limitations in solving ODEs. Additionally, rather than solely focusing on performance benchmarks, we aim to compare how neural network-generated solutions align with traditional human-solving strategies, providing a deeper interpretability component. That way users can both get a solution to a differential equation and see the steps taken to solve it. That way, even if the solution is incorrect, the user can still find where the neural network went wrong through visual compute graphs.

 
## Ethical & Educational Considerations

Furthermore, our project seeks to go beyond academic research by making our findings accessible through an interactive web interface, ensuring that anyone can use the program upon publishing our results. We understand there is the risk of students misusing this program in ways that violate school policies, such as cheating and plagiarism, but the applicability of this program in helping students learn and understand differential equations outweighs the chances of misuse, as it provides a learning opportunity for people who may not have access to advanced calculators.


## Conclusion

In conclusion, our project expands on existing research by integrating pattern matching and PINN methodologies while prioritizing interpretability and accessibility. By providing visual compute graphs and a free web-based interface, we make neural network-based ODE solving both transparent and widely available. While misuse is a concern, the educational benefits like helping users understand and verify solutions outweigh the risks, making this a valuable tool for students and researchers alike.


## References

[^1]: Amini, S., Hashemi, A., Azizi, A., & Ebrahimi, H. (2023). *Solving differential equations with Deep Learning: A beginner’s guide*. arXiv preprint arXiv:2302.12260. [https://arxiv.org/abs/2302.12260]

[^2]: Zang, Y., Bao, G., Ye, X., & Zhou, H. (2020). *Weak adversarial networks for high-dimensional partial differential equations*. Neurocomputing, 399, 305-315. 
[https://www.sciencedirect.com/science/article/abs/pii/S0021999118305527]

[^3]: MathWorks. (n.d.). *Solve ODEs Using a Neural Network*. [https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html]

[^4]: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707. 
[https://www.sciencedirect.com/science/article/abs/pii/S0925231220301909]
