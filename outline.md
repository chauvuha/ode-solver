# Differential Equation Solver

## Project Scope  
Build and train a neural network (NN) to use pattern matching and Physics-Informed Neural Network (PINN) formalisms to predict compact approximations of Ordinary Differential Equations (ODEs).  

## Members  
- Eve Zeng  
- Chau Vu  
- Dualeh<sup>2</sup>  
- Sultan Al Dawoodi  

## Introduction:

ODEs (ordinary differential equations) are an important way of modeling the world, showing up in different fields such as economics, biology, and physics. Being able to have accurate solutions to these equations gives power in analyzing complex systems. Our project focuses on testing a branch of neural network formalism called PINN (Physics informed neural networks) in its ability to solve differential equations, such that these equations have more interpretable solutions. This will allow more information to be gained from the solutions, and will also be clear and helpful to those who are looking for patterns and structures in these equations. We plan to develop a program based on previous research findings and target it towards a more pedagogical approach, hoping to help students understand ODE methods. Our final product will consist of a website hosting the trained and implemented Neural Network, descriptions, documentations, and methods on the theories behind our program and the mathematics. 


## Related Work:

Past literature in using neural networks to solve ODEs through various approaches. Some studies introduced Physics-Informed Neural networks (PINNs) as a strategy to solve first and second-order ODEs, and highlighted their potential applications in advancing physics simulations [Source 1]. Other studies built on such approaches. [Source 2] improved this method by embedding equation knowledge directly into the loss function, ensuring that the model learns to satisfy the actual equations rather than merely fitting data points while introducing tutorials and benchmark tests to evaluate performance. Meanwhile, MathWorks has presented an alternative approach that leverages neural networks to approximate ODE solutions in a closed analytic form, outlining a structured training process that includes data generation, network definition, and custom loss functions [Source 4]. Other research has extended deep learning techniques to solve partial differential equations (PDEs), training neural networks on randomly sampled time and space points to approximate solutions for cases where no exact analytical solutions exist [Source 3,5]. These works collectively demonstrate the growing potential of deep learning in solving differential equations. 

Our project distinguishes itself from existing research in neural network-based differential equation solvers through its focus on both pattern matching and Physics-Informed Neural Network (PINN) methodologies. While prior work has explored neural networks for solving ODEs, much of it has concentrated on either direct function approximation or strictly physics-informed approaches. Our approach integrates both perspectives, allowing us to evaluate their respective strengths and limitations in solving ODEs. Additionally, rather than solely focusing on performance benchmarks, we aim to compare how neural network-generated solutions align with traditional human-solving strategies, providing a deeper interpretability component. That way users can both get a solution to a differential equation and see the steps taken to solve it. That way, even if the solution is incorrect, the user can still find where the neural network went wrong through visual compute graphs. Furthermore, our project seeks to go beyond academic research by making our findings accessible through an interactive web interface, ensuring that anyone can use the program upon publishing our results. We understand there is the risk of students misusing this program in ways that violate school policies, such as cheating and plagiarism, but the applicability of this program in helping students learn and understand differential equations outweighs the chances of misuse, as it provides a learning opportunity for people who may not have access to advanced calculators. 

In conclusion, our project expands on existing research by integrating pattern matching and PINN methodologies while prioritizing interpretability and accessibility. By providing visual compute graphs and a free web-based interface, we make neural network-based ODE solving both transparent and widely available. While misuse is a concern, the educational benefits like helping users understand and verify solutions outweigh the risks, making this a valuable tool for students and researchers alike.


