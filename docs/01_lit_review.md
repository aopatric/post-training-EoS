# Some Literature Review

This Markdown doc will contain some of the basic lit review done for this project. Starting by organizing notes/topics in a hierarchical order.

Some papers I came across with contributions, roughly chronologically:

1. Wu, Ma & E (2018) "How SGD Selects the Global Minima in Over-parametrized Learning: A Dynamical Stability Perspective"
    - Perform analysis on the local minima obtained by *both* **batch** and **stochastic** GD approaches; find some restrictions/necessary conditions for a local minima to be "considered" by these algorithms. Namely, for sharpness $\lambda_\text{max}(H) = s$:
        - For **batch** GD, it is necessary that $s \leq 2 / \eta$; that is, GD can only converge to minima whose largest Hessian eigenvalue does not exceed $2 / \eta$.
        - For **stochastic** GD, it is necessary that the above property is satisfied but authors also introduce a *new* concept of **non-uniformity** (think, how *spread out* is the loss relative to the data?); authors find that for SGD to consider a local minima it must also have $\text{non-uniformity} \lesssim \sqrt{\frac{B}{\eta}}$ for a batch size $B << n$ (with dataset size $n$).
    - Although authors don't name the phenomena and attribute it to an abundance of sharp minima; they argue an optimizer stable for both *sharp* and *flat* minima will often find sharp ones when they are so abundant.
2. Cohen et al. (2021) "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"
    - Introduces the notion of EoS as it has become widely accepted. Authors' main claim is that **batch** GD, even with step sizes that violate classical stability limits (i.e., $\eta \lambda_\text{max} > 2$), often manages to avoid divergence when applied to deep nets. Instead, batch GD seems to enter a non-monotonic improvement regime where the *loss* continues to decrease despite local instability and the sharpness hovers around $\lambda_\text{max} \approx 2  / \eta$.
    - Characterize EoS with a few observations:
        - First, they (despite not naming it) implicitly discover the *progressive sharpening* phenomena (top Hessian eigenvalue increases durign early training and stabilizes approximately equal to $2 / \eta$). It is beyond this point that we see the EoS.
        - Second, they find that in the EoS regime, loss decrease is *non-monotonic*, gradient norms *spike*, and curvature *fluctuates*. Authors interpret this regime as a weakly unstable regime where the individual GD steps are unstable in that they overshoot true progress but the learning dynamics as a whole remain globally convergent.
        - Third, the authors provide a note on the failure of local stability theory to describe the global training dynamics of nonlinear models. That is, the $\eta \lambda_\text{max} < 2$ threshold comes from our local approximation of the loss surface as a quadratic, which seems to break down in EoS.
    - Show the phenomenon is consistent when using MLPs, CNNs, ResNets, VGGs on multiple datasets. Show the stability threshold is dependent on learning rate, argue this implies that the EoS is a generic attractor of gradient-based training dynamics.
3.  <!-- TODO: continue -->