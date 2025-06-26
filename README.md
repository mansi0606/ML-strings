# ML-strings
this is my btech project
Overview
This repository explores the application of Machine Learning (ML) techniques to problems in the string landscape. The string landscape refers to the vast array of possible Effective Fields Theories arising from compactifications of string theory. The string landscape, while rich in theoretical possibilities, presents a formidable challenge: its scale and complexity make traditional methods of exploration inefficient, if not infeasible. With its combinatorial complexity, it poses challenges ideal for ML, particularly in optimisation, classification, and pattern recognition.

This GitHub repository accompanies lectures held at the Avogadro Meeting 2024 in Naples, Italy, on December 18, 2024, by Stefano Lanza and Andreas Schachner. These lectures aim to bridge the gap between theoretical physics and modern ML, providing tools and tutorials for PhD students and researchers eager to apply ML methods to this frontier.

Lecture Summary
These lectures aim to provide a comprehensive overview of the emerging interplay between machine learning and string theory, focusing on applications to the string landscape. Over the past decade, ML has become an indispensable tool in a wide range of scientific disciplines, leveraging large datasets and high-dimensional spaces to identify hidden structures, optimise functions, and make predictions. These capabilities are particularly well-suited to the challenges posed by string theory, where datasets are often large and intricate, and the underlying physics can be highly non-linear and multi-faceted.

The topics covered in this lecture are as follows:

String Landscape Exploration: Tools to navigate and visualise the String Landscape.
Optimisation Algorithms: Implementations of ML models for searching specific vacua or configurations.
Pattern Recognition: Applications of neural networks to classify and identify symmetries or anomalies.
Data-Driven Insights: Methods for interpreting large datasets arising from string theory models.
Customisable Workflows: Easily adapt ML pipelines for specific physics questions.
By highlighting the above key results and methodologies, we hope our lectures will serve as a gateway for physicists and computer scientists interested in contributing to this interdisciplinary frontier. This GitHub repository will offer accessible code, examples, and interactive notebooks to enable readers to engage with the concepts hands-on. The repository will include implementations of ML models applied to example problems from string theory, fostering an interactive learning experience and encouraging further innovation.

Example 1: Toy model for inflationary regions
Here we present a simple example of supervised learning problem. The aim of problem is trying to check whether some members of a family of effective field theories can host slow-roll inflation, and how to find the field space region accommodating for inflation. For this example, we set up a binary classification problem in the notebook Toy_inflationary_models.ipynb: we label regions in field space depending on whether a given point allows for slow-roll inflation.

Example 2: Flux vacua on the rigid Calabi-Yau
As a second application, we study flux vacua on the rigid Calabi-Yau manifold. Flux compactifications on rigid Calabi-Yau manifolds offer a simplified yet rich framework for stabilising moduli and exploring effective 4D physics. They are particularly valuable for isolating the impact of fluxes and for providing a tractable subset of the string landscape suitable for testing new machine learning implementations.

The notebook ML_for_flux_vacua.ipynb introduces and covers various different aspects of working with ML and related data science or optimisation methods:

Getting Started: Introduction to ML concepts and their relevance to the String Landscape.
Data Preparation: Generating and preprocessing datasets from string theory models.
Dimensional Reduction: Techniques for dimensionally reducing datasets to save memory and runtime.
Visualisation: Present standard tools to visualise higher-dimensional datasets etc.
Benchmarking Models: Brief examples of how to benchmark ML architectures.
Model Training: Using NNs to explore vacua distributions.
Hyperparmeter optimisation: Undertaking a process of optimising hyperparameters.
Search optimisation problems: Employing Genetic Algorithms to find preferred flux vacua.
Resources
Given the limited time to discuss ML methods in the lecture, here are a couple of highly recommended books, lectures and reviews:

Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron
Probabilistic Machine Learning: An Introduction by Kevin Patrick Murphy
Lectures at Amsterdam University on jax and ML in Physics
Review on Data science applications to string theory by Fabian Ruehle
Lectures on Naturalness, String Landscape and Multiverse: A Modern Introduction with Exercises by Arthur Hebecker
Requirements
Python 3.8+
Key libraries: numpy, scipy, matplotlib, seaborn
Libraries from the JAX ecosystem: jax, jaxlib, flax, optax
Installation
Clone the repository and install dependencies:

$ git clone https://github.com/AndreasSchachner/ml-string-landscape.git
$ cd ml-string-landscape
$ pip install -r requirements.txt
