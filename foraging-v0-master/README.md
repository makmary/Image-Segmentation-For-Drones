# Foraging environments for the Gym toolkit

This package contains environments for playing the single- and multi-agent
games of foraging. The task of the agent(s) is to find food (fixed(or randomly) distributed checkpoints) in a confined 3D space in a minimum timeframe.

## Installation

To install the package, cd to the root directory of the package and run:

```
pip install -e .
```

## Usage

The environment acts as a regular gym environment. Call `reset()` to reset the environment. Call `step()` to perform one step.

### Initialization

When instantating the environment, following parameters can be used:

* `num_agents`: Number of agents. Note that the number of targets is equal to the number of agents;
* `map_size`: 3-element vector containg the size of the bounding box _[width, length, height]_.
* `fixed_goals`: An array of 3-element vectors, containing fixed initial positions of the goal cells. The environment will pull values from this array as long as they exist, otherwise the goals will be generated randomly. Note that the number of goals is the same as the number of agents. For instance, if this vector contains just two oints, two goals will be generated with fixed positions, the rest will be distributed randomly in the space.

### Observation space

Observation is a vector containing the current positions of the agents in the form _[[x1, y1, z1], ..., [xn, yn, zn]]_, where n is the number of agents.

### Action space

Action is a vector of discrete int values from _[0; 5]_ in the form of _[a1, .., an]_, where:

```
0 = move right;
1 = move left;
2 = move forward;
3 = move backward;
4 = move up;
5 = move down;
6 = do nothing.
```

### Reward

The agent receives -1 for each action and 0 if the target was found. The rewards are returned simulateniously as a vector of from _[r1, .., rn]_.
