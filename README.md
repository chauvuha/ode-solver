# Differential Equation Solver

## Project Scope  
Build and train a neural network (NN) to use pattern matching and Physics-Informed Neural Network (PINN) formalisms to predict compact approximations of Ordinary Differential Equations (ODEs).  

## Members  
- Eve Zeng  
- Chau Vu  
- Dualeh<sup>2</sup>  
- Sultan Al Dawoodi  

## Outline  
We are creating and training a neural network to solve differential equations. Being able to solve ODEs is valuable for modeling complex systems in mathematics, physics, and various other fields. The use of NN in solving differential equations is an ongoing research effort with connections to cutting-edge advancements. Differential equations rely heavily on pattern matching, and some have no analytical solutions. NNs can help find the most accurate and simple solution relevant to the problem, potentially providing either an analytical solution or a simple and accurate approximation.  

We are basing our research on several papers with different NN implementations. Our goal is to train and compare these models to determine the most optimal one. Additionally, we aim to create an interactive web interface that makes our findings accessible to the public. To accomplish this, we will first analyze traditional human approaches to solving ODEs, then evaluate how different NN implementations align with these methods. We will also test and fine-tune the models as needed to achieve good results.  

In general, we are interested in further applications of solving differential equations using NNs. What is the complexity limit of equations that can be solved? How would solving PDEs compare to solving ODEs?  

---

## Ethical Sweep  

### General Questions  

**Should we even be doing this?**  
Yes, this project would be highly beneficial to the math and science communities. Solving differential equations can be time-consuming, but a trained neural network could generate solutions much faster.  

**What might be the accuracy of a simple non-ML alternative?**  
Step approximations and manually solving equations are common non-ML alternatives.  

**What processes will we use to handle appeals/mistakes?**  
We can validate our solutions by converting them back into differential equations and comparing values for accuracy.  

**How diverse is our team?**  
Our team includes individuals from CS, Math, and Physics, as well as members from diverse racial and socioeconomic backgrounds, bringing different perspectives and opinions.  

### Data Questions  

**Is our data valid for its intended use?**  
Yes, we plan to use differential equation datasets that have been validated, labeled, and analytically proven to be correct.  

**What bias could be in our data?**  
Certain equations may be overrepresented due to their importance in specific research fields, leading to a bias in the dataset. Less commonly studied equations might have fewer validated solutions.  

**How could we minimize bias in our data and model?**  
We can collect and test our models with a diverse and representative dataset, incorporating equations used in research across different fields.  

**How should we audit our code and data?**  
We can graph results against input values to verify correctness and implement error or warning messages when discrepancies occur.  

### Impact Questions  

**Do we expect different error rates for different sub-groups in the data?**  
Yes, due to dataset biases, different error rates may emerge based on the type of equations, their applications, and the fields they originate from.  

**What are likely misinterpretations of the results, and how can we prevent them?**  
Users should understand that NN-generated solutions are not always correct and may not be optimal. To prevent misinterpretations, we can include user warnings and visual comparisons between NN outputs and expected results.  

**How might we impinge on individuals’ privacy and/or anonymity?**  
Our model will be trained solely on differential equation data, meaning no human subjects will be involved, ensuring privacy is not a concern.  
