#set document(
    title: [`tinysails`: Teaching an agent to sail],
    author: "Cesare Siringo (cesare.siringo@santannapisa.it)",
)
#set page(
    paper: "a4"
)
#set heading(numbering: "1.a")

#title()

= Introduction
The aim of this project is training a small neural networks to sail a regatta in minimal time. Here, a regatta is a collection of buoys that must be reached in order. The complexity of this task is not readily apparent, but it boils down to solving the biggest problem it presents: sailing upwind. To do that an agent must learn not to aim directly at the next buoy, but to approach it at an angle. To do that it must forego the immediate reward of closing in on the buoy as fast as possible and reap it afterwards on the closing leg instead. In #ref(<phys>) we explain how the physical model of the boat is constructed and what assumptions are made. In #ref(<env>) we explain the main choices made in structuring the environment and decisions that apply generally to every architecture considered. In #ref(<train>) we show the performance of all architectures in order of increasing complexity. The Proximal Policy Optimization architecture was also implemented, but left out as it did not provide benefits, probably as it disincentivises exploration and is more subject to hyperparameter variation.

= Physical Modeling <phys>
Modeling the physics of a real boat would be extremely complicated and computationally expensive, as it requires multiphase fluid simulations at every step. Here, the model is heavily simplified while still retaining most of the qualitative features of a real sailboat.

== Assumptions
We consider a 2-dimensional sailboat with three surfaces: the main sail, a rudder, and a centerboard. The latter is considered of infinite surface as to assume no lateral movement can happen.
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

Two simple constant are hardcoded:
- $I = 1/2m$: rotational inertia
- $d_r = 1$: rudder distance to CoG

To find kinematics, simple explicit euler integration is employed.

=== Modeling the sail and centreboard
First, we extract relative wind:
$w_(r e l) = w - hat(h)v$

Given a sail versor $hat(s)$, we find the wind's perpendicular component to it: $w_perp = w_(r e l) - (w_(r e l) dot hat(s))hat(s)$ and the sail's force on the hull: $F_s = L_s w_perp$ //TODO: w_perp squared?

We can now use our centreboat assumption project this force on the boat's heading versor:

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
The environment is comprised of a boat and a sequence of buoys. Buoys must be reached in order in order to be considered valid. A buoy if reached if the boat has passed over it. As unrealistic as this is, it is a good objective since it is harder to achieve than simpling passing around a buoy and it is also easier to check whether or not a buoy has been reached.

The state space is inherently continuous. An attempt has been made to discretize it with kohonen networks (see `python -m demos.kohonen_boat`), but it is clear that in order to capture significant information there would need to be hundreds or thousands of neurons, rendering Monte-Carlo, TD-learning and Q-learning unachievable.

As a result, we shifted to their continuous counterparts: REINFORCE and Actor-Critic, while keeping the action space discrete. Every model has three output neurons, that corresponds to the rudder being at $-45deg$, $0deg$, $+45deg$.

= Training the Agent <train>
== REINFORCE
== Actor Critic
= Conclusions and Further Work