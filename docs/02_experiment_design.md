<!-- NOTE: lit review is giving me a headache so im working on this instead -->

# Experiment Design

Approaching this by taking a look at how some of the other papers in this niche have gone about testing the emergence of this.

From two sides, need to understand: how we typically classify what is EoS and what isn't (what plots, etc.) as well as how we typically analyze training dynamics in post-training (typical setups, toy settings, etc.); the experiment suite for this project will be a derivative of these two categories.

### Testing for the Edge of Stability

Some common themes present across experiments:
- **Model Architectures, Tasks**: EoS has been observed across a variety of architectures and datasets in the existing literature. Cohen et al. (2021) provides a demonstraction of EoS using a *fully-connected* MLP (2 hidden layers, width 200, tanh activation) on CIFAR-10, *CNNs* (ResNet-32, VGG-19) on CIFAR-10, and a small *transformer* on WikiText-2 language modeling tasks.
- **Commonly Tracked Metrics**: As a rule of thumb, the primary metric for studying EoS is exactly the *largest Hessian eigenvalue* $\lambda_\text{max}$. This can be done every $x$ amount of steps during training to avoid overloading the computation.
    - Note: <!-- NOTE: come back to this and add how I can compute an approximation / efficiently get this lambda_max. -->
- **Triggers for EoS**: Need to be able to reproduce the conditions to observe EoS fairly handily; key requirements from the literature seems to be a sufficiently large *learning rate* such that the sharpness is able to saturate and reach $2 / \eta$; otherwise the sharpness may be too low and we might undershoot EoS. Literature claims we should reach EoS for any 'reasonable' learning rate when using batch GD.

- **Common Visualizations, Plots**: Can typically get a strong picture of EoS dynamics with a few key plots, namely *sharpness vs. iterations* and *train loss vs. iterations* each at varied learning rates in some fixed set used for sweeping $\eta$. Should see the former reach $\approx 2 / \eta$ and hover,  should see the latter drop then reach a non-monotonic region coinciding  with EoS.

### Post-Training Paradigms and Problem Settings

The other side of the gap is the post-training space; for the purpose of making things clearer, a formal definition of post-training to be used in the project:

> A formal definition of post-training:

For a given pre-trained set of parameters $\theta_\text{pre}$ trained with (general) objective $\mathcal L_\text{pre}(\theta)$, a **post-training phase** using batch gradient descent consists of additional training updates
$$
\theta_\text{post} = \theta_\text{pre} - \sum_{t=1}^T \eta \cdot \nabla \mathcal L_\text{post}(\theta_t)
$$
where $\mathcal L_\text{post}$ is a new loss function possibly defined over a data distribution independent of the previous or using a novel feedback mechanism.

With this in mind, positioning to study *three domains*:
- *Supervised Fine Tuning* (SFT) $\to$ defining this as full-parameter (not necessarily full-layer) fine-tuning.
- *Parameter-Efficient Adaptation* (e.g. LoRA) $\to$ defining this as using LoRA adapters to minimize the amount of updated parameters in fine-tuning.
- *Reward-Based Optimization* (e.g. RLHF, DPO, generally RLVR) $\to$ self-explanatory, the RL-based post-training category of methods.

Each probem setting involves a target objective, novel curvature regimes and loss landscapes, often times lower learning rates employed in practice, and often require new balances in generalization/retention from pre-training.

### Designing Experiments for the Edge of Post-Training Stability

Given the above, a suite of experiments is outlined by domain below:

1. **Supervised Fine Tuning (SFT)**
    - Fine-tuning a pre-trained LM (for size, using GPT-2 Medium) using Alpaca for instruction folowing. Using various LRs, tracking the metrics described above (train loss, sharpness vs. iterations). Using SGD and combining knowledge from both EoSS and Transformer-based EoS stuff.
    - Goal with this set of runs is to plot sharpness vs. step, the value $\eta \cdot \lambda_\text{max}$ vs. step, training loss, and retention on pretraining corpus (planning on using WikiText2).
2. **PEFT (LoRA)**
    - To get some results when we *don't* modify full parameters, can run a similar setup using the *same* dataset and model as the full SFT setting; changes would be using LoRA adapters of varying rank as well as varying learning rates as before.
    - Target visualizations in these would be something like overlaying sharpness trajectories, mostly the same visualizations as full-layer SFT but offers (potentially) different dynamics.
3. **RLVR**
    - Definitely the hardest to plan around, but a good toy problem setting could be a synthetic arithmetic dataset. Consider doing something like generating a few types of problems for ~10k samples. Then can wrap the dataset in a `gymnasium`-like API and set up a RLVR loop with an existing PPO trainer.

<!--- TODO: add more detail, if time permits -->
