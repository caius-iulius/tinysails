#set document(
    title: [`tinysails`: Teaching an agent to sail],
    author: "Cesare Siringo",
)
#set page(
    paper: "a4"
)
#set heading(numbering: "1.1.a")
#set text(12pt)

#title()
Cesare Siringo `cesare.siringo@santannapisa.it`

#align(center, image("thumbnail.png", width: 75%))

= Introduction
The aim of this project is training a small neural network to sail a regatta in minimal time. Here, a regatta is a collection of buoys that must be reached in order. The complexity of this task is not readily apparent, but it boils down to solving the biggest problem it presents: sailing upwind. To achieve that an agent must learn not to aim directly at the next buoy, but to approach it at an angle and it must forego the immediate reward of closing in on the buoy as fast as possible and reap it afterwards on the closing leg instead. In #ref(<phys>) we explain how the physical model of the boat is constructed and what assumptions are made. In #ref(<env>) we explain the main choices made in structuring the environment and decisions that apply generally to every architecture considered. In #ref(<train>) we show the performance of all architectures in order of increasing complexity. The Proximal Policy Optimization architecture was also implemented, but left out as it did not provide benefits, probably as it disincentivises exploration and is more subject to hyperparameter variation.

= Physical Modeling <phys>
Modeling the physics of a real boat would be extremely complicated and computationally expensive, as it requires multiphase fluid simulations at every step. Here, the model is heavily simplified while still retaining most of the qualitative features of a real sailboat.

== Assumptions
We consider a 2-dimensional sailboat with three surfaces: the main sail, a rudder, and a centerboard. The latter is assumed to have an infinite surface area so that no lateral movement occurs.
Every surface is modeled as a "fluid reflector", that means that when the fluid impacts it with a force, the surface extracts the perpendicular component, while ignoring the parallel component, essentially projecting the force on its axis. This is different from real sails as they show wing-like behavior especially in upwind sailing.

The boat state is defined by the following variables:
- $m$: mass
- $C_b$: boat drag coefficient
- $C_r$: boat rotational drag coefficient
- $L_s$: sail lift coefficient
- $L_r$: rudder lift coefficient
- $arrow(x)$: boat position
- $hat(h)$: boat heading versor
- $v$: boat speed (scalar due to lateral movement assumption)
- $omega$: boat rotational velocity

Two simple constants are hardcoded:
- $I = 1/2m$: rotational inertia
- $d_r = 1$: rudder distance to CoG

To find kinematics, simple explicit euler integration is employed.

=== Modeling the sail and centerboard
First, we extract relative wind:
$w_(r e l) = w - hat(h)v$

Given a sail versor $hat(s)$, we find the wind's perpendicular component to it:

$w_perp = w_(r e l) - (w_(r e l) dot hat(s))hat(s)$

and the sail's force on the hull:

$F_s = L_s w_perp$ //TODO: w_perp squared?

We can now use our centerboard assumption project this force on the boat's heading versor:

$F_a = (hat(h) dot F_s)hat(h)$

It is simple to maximize this formula and find a closed formula for the optimal sail angle:

$theta_w = theta(w_(r e l)) \ theta_b = theta(-hat(h)) \ alpha = (theta_w - theta_b) / 2$

The aerodynamic force then becomes:

$1/2 L_s (||w_(r e l)|| + w_(r e l) dot hat(h))$

=== Modeling the rudder
Given a rudder angle $phi$, we apply the same math as the sail to find the rudder's drag on the boat and the induced rotational acceleration.

$alpha_r = -1/2 (C_r d_r sin(2 phi) v)/ I \ F_r = C_r sin(phi)^2 v$

=== Drags
Linear drag and rotational drag can be expressed as:

$F_d = C_d v^2$

$M_d = C_r omega^2$

=== Validating the physical model
To show that the model captures the relevant qualitative characteristics, we plot the polar diagram of the boat at varying wind speeds.
In the context of sailing, a polar diagram is a tool used by sailors to estimate the expected speeds at different wind angles. The following images show a striking similarity of the model to empirical data.

#grid(
    columns: 2,
    figure(
    image("model_polar_diagram.png", width: 80%),
    caption: [
        Polar diagram produced by `python -m demos.boat_model`
    ],
    ),
    figure(
    image("real_polar_diagram.png", width: 80%),
    caption: [
        Example of a real polar diagram
    ],
    )
)

As a result, we can make some predictions about the optimal strategy:
- sailing upwind is impossible, an optimal angle must be found.
- sailing directly downwind is less efficient than sailing at a small angle (this is due to speed being dependent on relative wind), a good model should capture that.

= Environment Architecture <env>
The environment is comprised of a boat and a sequence of buoys. Buoys must be reached in order in order to be considered valid. A buoy is reached if the boat has passed over it. As unrealistic as this is, it is a good objective since it is harder to achieve than simply passing around a buoy and it is also easier to check whether or not a buoy has been reached.

The state space is inherently continuous. An attempt has been made to discretize it with kohonen networks (see `python -m demos.kohonen_boat`), but it is clear that in order to capture significant information there would need to be hundreds or thousands of neurons, rendering Monte-Carlo, TD-learning and Q-learning unachievable.

As a result, we shifted to their continuous counterparts: REINFORCE and Actor-Critic, while keeping the action space discrete. Every model has three output neurons, that corresponds to the rudder being at $-45deg$, $0deg$, $+45deg$.

The state returned by the environment is a 7-tuple representing the distance to the next buoy, sine and cosine of the angle to the next buoy, sine and cosine of relative wind, boat speed, boat rotational velocity.

= Training the Agent <train>
Since training a RL agent requires both dense and sparse rewards, all loss functions used present three elements: Velocity Made Good as sparse reward, a sparse reward for reaching a buoy, and a punishment for losing time.
Some hyperparameters are shared amongst all considered models. These are:
- `tanh` activation function for all layers
- a $Delta t$ of $0.1$ for euler integration during training
- a returns normalization constant of `1e-9`
- usage of the `Adam` optimizer
- a discount factor of $gamma = 0.99$

== REINFORCE
Training script: `python train_reinforce_small.py`

The REINFORCE training algorithm is the simplest continuous state space method possible. It is essentially the continuous analog of the Monte-Carlo method. We trained a network on a simple 4-buoy which tested all gaits except tacking, to establish a baseline.

The specific hyperparameters are:
- network of topology $7 times 8 times 8 times 3$
- learning rate of `0.002`
- 2000 episodes, with a time limit of 1000 ticks each

The reward function is a combination of sparse and dense rewards. The sparse reward is engineered in such a way to incentivize passing the buoy as soon as possible.

$ r = (arrow(v)_"boat" dot hat(u)_"buoy") Delta t + cases(
  5000 / (t - t_"last_buoy") & "if" d < r_"buoy",
  0 & "otherwise"
) $

The resulting network achieved stable sub-400-tick completion time after less than 2000 training episodes. However, the network showed jittery controls, forgetting the existence of the $0deg$ output neuron, opting to bang the rudder instead. We observe the disappearance of this phenomenon with the bigger models tested later. This has led us to conclude that the network is too small to learn the dynamics of angular momentum.

We also trained a network with hidden layers of 16 neurons. But observed the inability of the REINFORCE method to handle bigger networks.

Shown below are the smoothed rewards and completion times of a big and a small network, with the original data shown faded.

#figure(image("reinforce_comparison.png"), caption: [Reward and time to completion of a small and a big network trained using REINFORCE])

== Actor Critic
We now move to the Actor-Critc method, which, instead, can be viewed as the continuous analogue of TD-learning. There are many variants of Actor-Critic with different features. For simplicity, we focus on one-step Actor-Critic and Batched Actor-Critic.

The graph below shows the performance of REINFORCE and Actor-Critic (as structured in #ref(<cac>)) trained on the same course. It can be seen that Actor-Critic is clearly more sample-efficient than REINFORCE, finding acceptable solutions in around 10 episodes, while also being more locally stable than REINFORCE (that is, while the reward function still fluctuates, it does so in a bigger timeframe).

#figure(image("reinforce_vs_actorcritic_oldcourse.png"), caption: [REINFORCE vs Actor-Critic trained on the same course])

=== One-Step Actor Critic <cac>
Training script: `python train_actorcritic.py`

The model's structure is comprised of a shared feature extractor of topology $7 times 32 times 32 times 32$, an Actor head of three neurons, and a Critic head of one neuron.

As the complexity of the training method increases, so do its performance fluctuations on hyperparameters. In this version of the network we introduced a learning schedule with `pytorch`'s builtin cosine annealing schedule and a linearly decreasing entropy schedule, to prevent the network from immediately collapsing on the reward of an intermediate step.

Hyperparameters:
- Cosine annealing learning rate from `5e-4` to `5e-5`
- Linearly decaying entropy from `0.1` to `0.005`
- 2000 episodes, with a time limit of 1500 ticks each

The reward function has been tweaked as follows (where $i$ is the current boat index):

$ r = ( 0.1(i - N + 1) + 0.5(arrow(v)_"boat" dot hat(u)_"buoy") ) Delta t + cases(
  50 & "if" d < r_"buoy",
  0 & "otherwise"
) $

This method has been tested on a course which tests all possible gaits, including tacking, and is capable of finding good policies which are both capable of basic tacking with slow turns (leading to occasional failure in irons), and stable controls with no rudder banging. However it still suffers from occasional policy collapse, which sometimes it recovers from, and must be run multiple times to find close-to-optimal solutions.

#figure(image("actorcritic_comparison.png"), caption: [One-Step Actor-Critic performance])

=== Batched Actor Critic <bac>
Training script: `python train_actorcritic_batched.py`

To solve reward stability and policy collapse, a form of batching was introduced; a model records the activation of $N$ steps in the environment before running backprop on them as a whole, stabilizing the gradient ascent. This also means that we can take advantage of the efficient tensor backprop built into pytorch. On the other hand, more training is needed since this method chases rewards less greedily.

The model's structure is unchanged, while the hyperparameters are as follows:
- Batch size of 32
- Cosine annealing learning rate from `3e-3` to `1e-5`
- Linearly decaying entropy from $0.1$ to $0$
- Critic loss scaling factor of `0.5`
- Gradient clipping to `0.5`
- 2000 episodes, with a time limit of 1500 ticks each

The reward function is unchanged.

#figure(image("actorcritic_batched_comparison.png"), caption: [Batched Actor-Critic performance])

While being a great improvement over One-Step Actor-Critic, the model still suffers from some instabilities and makes some clearly suboptimal choices during inference. Moreover, when tested on other courses, it shows some signs of overfitting, while still being able to complete generic courses.

=== Randomized Environments <rac>
Training script: `python train_actorcritic_randomized.py`

To provide resistance to overfitting, the model is trained on random 7-buoy environments, where each buoy must have a distance to the previous buoy of at least 30 units.

The hyperparameters are unchanged. Since course difficulty varies randomly, the introduction of an hardness coefficient to the reward function was considered. However, it yielded counterproductive results. This was attributed to the fact that a harder course (say, because of increased distance to the next buoy), already had a bigger opportunity to reap continuous rewards, so the hardness of a course is already encoded in the reward function. We opted for a bigger weighting to the continuous reward instead.

$ r = ( 0.2(i - N + 1) + 0.5(arrow(v)_"boat" dot hat(u)_"buoy") ) Delta t + cases(
  50 & "if" d < r_"buoy",
  0 & "otherwise"
) $

In order to extract a useful performance metric for the model, it is run over the same standardized course used in the non-randomized cases. These values aren't used during training, but are useful for visual comparisons with the previous methods.

#figure(image("actorcritic_randomized_comparison.png"), caption: [Randomized Actor-Critic performance])

Below is the comparison between all three Actor-Critic methods. The interesting result is that even though the randomized method isn't trained on the evaluation course, it is both better and more stable on it than the previous, specialized, models.

#figure(image("actorcritic_comparison_all.png"), caption: [Comparison of all Actor-Critic methods])

= Conclusions and Further Work