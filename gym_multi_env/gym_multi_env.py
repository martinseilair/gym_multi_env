import numpy as np
from gym import core, spaces
from gym.utils import seeding
import gym

class GymMultiEnv(core.Env):

    def __init__(self, domain_names, reward_mix=None, join_spaces=True):
        self.envs = []
        self.n_envs = len(domain_names)
        self.action_space_joined = False
        self.observation_space_joined = False
        self.domain_names = domain_names

        self.envs = [gym.make(domain_name) for domain_name in domain_names]

        if join_spaces:
            # try to join action spaces

            # action space
            if len(set([type(env.action_space) for env in self.envs])) == 1:
                action_space, action_mux = self.space_cat([env.action_space for env in self.envs])
                if not action_space is None:
                    self.action_space = action_space
                    self.action_mux = action_mux
                    self.action_space_joined = True

            # observation space
            if len(set([type(env.observation_space) for env in self.envs])) == 1:
                observation_space, observation_mux = self.space_cat([env.observation_space for env in self.envs])
                if not observation_space is None:
                    self.observation_space = observation_space
                    self.observation_mux = observation_mux
                    self.observation_space_joined = True

        # if join failed or not wanted: pack as tuple
        if not self.action_space_joined:
            self.action_space = spaces.Tuple([env.action_space for env in self.envs])

        if not self.observation_space_joined:
            self.observation_space = spaces.Tuple([env.observation_space for env in self.envs])

        if reward_mix == None:
            self.reward_mix = np.ones(self.n_envs)
        else:
            self.reward_mix = reward_mix

        self.seed()
        self.reset()

    def space_cat(self, space_list):
        out_space = None
        space_mux = None
        if type(space_list[0]) == gym.spaces.discrete.Discrete:
            space_mux = np.ones(len(space_list), dtype=np.int32)
            out_space = spaces.MultiDiscrete([space.n for space in space_list])
        elif type(space_list[0]) == gym.spaces.box.Box:
            space_mux = [space.shape[0] for space in space_list]
            high = np.concatenate([space.high for space in space_list])
            low = np.concatenate([space.low for space in space_list])
            out_space = spaces.Box(low, high, dtype=np.float32)

        return out_space, space_mux

    def info(self):
        p = 20
        print("Domains:", self.n_envs)
        for i, env in enumerate(self.envs):
            print((str(i+1)+'.').ljust(5),self.domain_names[i].ljust(p), str(env.action_space).ljust(p), str(env.observation_space))
        print("Action space".ljust(20), self.action_space)
        print("Observation space".ljust(20), self.observation_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        for env in self.envs:
            env.seed(self.np_random.randint(0, 9223372036854775807))
        return [seed]

    def convert_observation(self, obss):
        if self.observation_space_joined:
            npobss = np.array(obss)
            if type(self.observation_space) == gym.spaces.box.Box:
                return np.concatenate(np.array(npobss))
            elif type(self.observation_space) == gym.spaces.multi_discrete.MultiDiscrete:
                return np.array(npobss)
            else:
                return tuple(obss)
        else:
            return tuple(obss)

    def convert_action(self, action):

        if self.action_space_joined:
            if type(self.action_space) == gym.spaces.box.Box:
                p = 0
                a = []
                for ad in self.action_mux:
                    a.append(action[p:p + ad])
                    p = p + ad
                return a
            elif type(self.action_space) == gym.spaces.multi_discrete.MultiDiscrete:
                return action
            else:
                return action
        else:
            return action

    def reset(self):
        obss = []
        for env in self.envs:
            obs = env.reset()
            obss.append(obs)
        return self.convert_observation(obss)

    def step(self, action):
        a = self.convert_action(action)
        obss = []
        self.rewards = []
        alldone = False
        for i, env in enumerate(self.envs):
            obs, reward, done, _ = env.step(a[i])
            alldone |= done
            obss.append(obs)
            self.rewards.append(reward)
        return self.convert_observation(obss), np.dot(self.reward_mix, self.rewards), alldone, {}

    def close(self):
        for env in self.envs:
            env.close()

    def reward(self):
        return self.rewards

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return tuple([env.render('rgb_array') for env in self.envs])
        elif mode is 'human':
            for env in self.envs:
                env.render()
        else:
            super(GymMultiEnv, self).render(mode=mode)