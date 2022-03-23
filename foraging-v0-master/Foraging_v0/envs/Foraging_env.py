import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms, animation
# from stl import mesh
from mpl_toolkits import mplot3d
import os
from sklearn.neighbors import NearestNeighbors


class ForagingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_pos=False, num_agents=1, num_obst=4, target=np.array([0., 0., 0.]),
                 map_size=np.array([[-2., -2., 0.], [2., 2., 2.]], dtype=np.float64)):
        """
            Description:
                Agents start the round with a specific initial configuration and zero speed. When a
                projectile approaches, agents can change their speed by any value in between. The task
                 is to avoid collision with the projectile.
            Observation:
                Type: list [np.array[pos, vel] x (num_agents + projectile)]


            Actions:
                Type: Box(3) * num_agents
                Num    Action                    Min            Max
                0      X Velocity                 - 5.0          5.0
                1      Y Velocity                 - 5.0          5.0
                2      Z Velocity                 -2.0           2.0

            Reward:
                 The reward decreases when the minimum radius is reached in inverse proportion to the
                  radius.
                 The reward decreases as one drone passes under another, maximum deduction at minimum
                  z distance.
                 After the stone stops, all non-colliding drones receive 100 points.
            Starting State:
                 Set by the user.
                 The initial speed is zero.
            Episode Termination:
                 The episode ends after the projectile stops or after violation of emergency radius.
            """
        if map_size is None:
            map_size = np.array([[0., 0., 0.], [10., 10., 10.]], dtype=np.float64)
        self.bb = map_size
        self.num_agents = num_agents
        self.num_obst = num_obst
        self.target = target
        self.action_space = [spaces.Box(
            low=np.array([-5.0, -5.0, -5.0]), high=np.array([5.0, 5.0, 5.0]), dtype=np.float64
        ) for i in range(self.num_agents)]

        if init_pos.any():
            self.state = init_pos
        else:
            # self.state = [np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64) for i in range(self.num_agents + 1)]
            self.state = np.zeros((self.num_agents + self.num_obst + 1, 3))

        self.reward = [0 for i in range(self.num_agents)]
        # self.done = [False for i in range(self.num_agents)]
        self.done = False
        self.emer_r = 0.2
        self.min_r = 0.2
        self.rate = 10
        self.a_max = 5
        self.v_max = 5

        # reward coef
        self.C = 10 # collision
        self.T = 20 # target

        # Plotting routine initialization
        self.fig = plt.figure()
        self.ax = mplot3d.Axes3D(self.fig)
        self.fig.add_axes(self.ax)
        self.ax.set_xlim3d(self.bb[0, 0] - 1, self.bb[1, 0] + 1)
        self.ax.set_ylim3d(self.bb[0, 1] - 1, self.bb[1, 1] + 1)
        self.ax.set_zlim3d(self.bb[0, 2] - 1, self.bb[1, 2] + 1)
        self.agent_poly = []
        self.goal_poly = []
        plt.ion()

        # Create a poly instance for each agent and each goal
        # self.mesh = mesh.Mesh.from_file(os.path.dirname(__file__) + '/mesh/quadcopter.stl')
        # for i in range(self.num_agents):
        #     self.agent_poly.append(mplot3d.art3d.Poly3DCollection(
        #         self.mesh.vectors*0.5+self.state[i][0]
        #     ))
        #     self.ax.add_collection(self.agent_poly[i])

        # self.agent_poly.append(mplot3d.art3d.Poly3DCollection(
        #     self.mesh.vectors * 0.5 + self.state[-1][0]
        # ))
        # self.agent_poly[-1].set_facecolor('r')
        # self.ax.add_collection(self.agent_poly[-1])
        # Reset the state
        self.reset(init_pos)

    # def animate(self, frame, trajectories):
    #     for i in range(self.num_agents+1):
    #         self.agent_poly[i].set_verts(self.mesh.vectors*0.5 + trajectories[i][frame])

    def reward_collisions(self, dist, labels):
        for i in range(self.num_agents):
            dist_sum = 0

            for j in range(dist[i].shape[0]):
                if dist[i][j] <= 0.2 and dist[i][j] != 0:
                    print(dist[i])
                    print('state ', self.state)
                    self.reward[i] = -2e4
                    self.done = True 

                # if dist[i][j] >= self.emer_r:
                #     dist_sum -= self.C / (dist[i][j] +1e-4)
                # elif dist[i][j] != 0:
                #     dist_sum = -1e4
                #     self.done = True
                #     break
            # self.reward[i] = dist_sum

    def reward_target(self):
        for i in range(self.num_agents):
            v = np.linalg.norm(self.target - self.state[i])
            if v > 0.4:
                # self.reward[i] += self.T / v
                self.reward[i] += 0
            else:
                print('Goal achieved!')
                self.reward[i] =1e4
                self.done = True

    def step(self, action):
        self.reward = [0 for i in range(self.num_agents)]
        self.actions = action
        self.target = self.state[-1]

        for i, a in zip(range(self.num_agents), action):
            self.state[i] = self.state[i] + a * 1 / self.rate
        dist, labels = get_neighbors(self.state[:-1, :2], self.num_agents + self.num_obst, self.min_r)
       
        self.reward_target()
        self.reward_collisions(dist, labels)

        if (self.state[0] > 3).any() or (self.state[0] < -3).any():
            
            self.done = True
            self.reward = [-2e4 for i in range(self.num_agents)]

        self.reward = [(i - 11) / (2 * 1e4) for i in self.reward]
        return self.state, self.reward, self.done

    def reset(self, init_pos):
        self.state = init_pos
        self.done = False

        for i in range(self.num_agents):
            self.reward[i] = 0

        # Reset meta dict
        self.prev_animation_state = self.state.copy()
        self.info = dict()

        return self.state

    def render(self, stone_cur_state):
        trajectories = []
        self.state[-1][0] = stone_cur_state
        for i in range(self.num_agents + 1):
            l1 = np.linspace(0, 1, 2)
            traj = (self.state[i][0].astype(np.float64) - self.prev_animation_state[i][0].astype(np.float64)) * l1[:, None]
            trajectories.append(self.prev_animation_state[i].astype(np.float64) + traj)

        # Hold the plot if all targets were acquired
        if np.all(self.done):
            hold = True
        else:
            hold = False

        for i in range(2):
            self.animate(i, trajectories)
            plt.show(block=hold)
            plt.pause(0.001)

        self.prev_animation_state = self.state.copy()


def get_neighbors(data, max_neighbours, min_r, algo='ball_tree'):
    # possible algo: 'auto', 'ball_tree', 'kd_tree', 'brute'
    # description: https://scikit-learn.org/stable/modules/neighbors.html

    nbrs = NearestNeighbors(n_neighbors=max_neighbours, algorithm=algo, radius=min_r).fit(data)
    dist, labels = nbrs.radius_neighbors(data)

    return dist, labels


if __name__ == '__main__':
    ep_len = 340
    # init_pos = [np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float64),
    #             np.array([[2, 2, 2], [0, 0, 0]], dtype=np.float64),
    #             np.array([[3, 3, 3], [0, 0, 0]], dtype=np.float64),
    #             np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)]  # stone
    # action = [np.array([3.0, 3.0, 4.0]), np.array([-4.0, -4.0, -4.0]), np.array([0.0, -4.0, -4.0])]

    init_pos = [np.array([[1, 1, 1], [0, 0, 0]]),
                np.array([[1.2, 1., 1.2], [0, 0, 0]]),
                np.array([[1.3, 1., 1.3], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]])]  # stone
    action = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]

    env = ForagingEnv(init_pos)
    env._max_episode_steps = ep_len
    state = env.step(action)
    print(env.state)
