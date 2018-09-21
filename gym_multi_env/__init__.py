from gym.envs.registration import register
import gym
import hashlib

def create(domain_names, reward_mix = None, join_spaces=True):
    # register environment
    prehash_id = str(domain_names)
    h = hashlib.md5(prehash_id.encode())
    gym_id = h.hexdigest()+'-v0'


    # avoid re-registering
    if gym_id not in gym_id_list:
        register(
            id=gym_id,
            entry_point='gym_multi_env.gym_multi_env:GymMultiEnv',
            kwargs={'domain_names': domain_names, 'reward_mix': reward_mix, 'join_spaces': join_spaces}
        )
    # add to gym id list
    gym_id_list.append(gym_id)

    # make the gym env
    return gym_id


gym_id_list = []