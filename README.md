## 🧠 Abstract

Sequential sampling theory stands as one of the most influential frameworks in cognitive psychology for explaining human decision-making processes.  
This theoretical approach models decision-making as a **stochastic process within bounded domains**, where the **first-passage time (FPT)** distribution serves as the model's probability function.

Despite their theoretical elegance and empirical success, sophisticated variants of sequential sampling models — with **time-dependent boundaries** and **space–time-dependent drift functions** — present significant computational challenges, primarily due to the difficulty in efficiently computing their FPT distributions.

Traditional numerical methods require computationally expensive recalculations for each parameter configuration, while recent machine learning approaches, though computationally efficient, often lack theoretical grounding in cognitive science.  

We introduce **Theory-Informed Neural Networks (TINNs)**, a novel approach that bridges this gap by **embedding cognitive theory directly into neural network architectures**.  
Drawing inspiration from *physics-informed neural networks (PINNs)*, our method incorporates the underlying mathematical structure of cognitive models — specifically, the **Kolmogorov equations governing diffusion processes** — into the learning objective, simultaneously optimizing numerical approximation and parameter estimation.

TINNs are demonstrated across three critical applications:

1. Numerical approximation of FPT distributions  
2. Parameter estimation using simulation data  
3. Empirical data fitting  

Comprehensive evaluations on various **time-dependent cognitive models** (urgency signal, Ornstein–Uhlenbeck, diffusion model for conflict, and collapsing boundary models) reveal that TINNs achieve comparable or superior accuracy to traditional numerical methods while maintaining the computational efficiency of machine learning approaches.

Moreover, TINNs exhibit **enhanced interpretability**, as their architecture explicitly encodes theoretical constraints.  
This work introduces a **mathematically rigorous yet computationally efficient framework** that opens new avenues for implementing complex, theoretically grounded computational models of cognition.
