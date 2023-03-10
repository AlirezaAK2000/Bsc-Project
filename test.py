from env.carla_env import *
import json
from agents.common.networks import ActorCriticPPOAtari, SACActorAtari, DeepQNetworkAtari
import torch as T
import random
from torch.utils.tensorboard import SummaryWriter

random.seed(1)

if __name__ == '__main__':
    episodes = 10

    with open("config.json", 'r') as f:
        conf = json.load(f)

    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    num_pedestrians = [0, 200, 500]
    

    run_name = f"test_{conf['test']['agent']}_{int(time.time())}"
    writer = SummaryWriter(f"{conf['test']['log_dir']}/{run_name}")
    
    for n in num_pedestrians:
        conf['carla_test']['NUM_PEDESTRIANS'] = n

        with CarlaEnv(conf['carla_test'], test=True) as env:

            agent = None
            if conf['test']['agent'] == 'sac':
                agent = SACActorAtari(env.n_action, env.frame_num).to(device)
                agent.load_state_dict(T.load('tmp/sac/sacsacSAC_discrete0.pth'))

            elif conf['test']['agent'] == 'ppo':
                agent = ActorCriticPPOAtari(env.n_action, has_continuous_action_space=False, action_std_init=0,
                                            device=device).to(device)
                agent.load_state_dict(T.load('tmp/ppo/PPO_carla_0_0.pth'))
            elif conf['test']['agent'] == 'dqn':
                agent = DeepQNetworkAtari(env.n_action).to(device)
                agent.load_checkpoint()

            agent.eval()


            def select_action(state):
                output = agent.forward(state)
                action = T.argmax(output)
                return action.item()


            for eps in range(episodes):
                state = env.reset()
                done = False
                score = 0
                speeds = []
                covered_dist = 0
                col_with_ped = 0
                print(f"episode {eps}")
                while not done:
                    state = T.tensor(state, dtype=T.float, device=device)
                    action = select_action(state)
                    state, reward, done, info = env.step(action)
                    speeds.append(sum(info['linear_speeds']) / len(info['linear_speeds']))
                    covered_dist = info['dist_covered']
                    col_with_ped = 1 if info['col_with_ped'] == 1 else col_with_ped
                    score += reward
                                    
                writer.add_scalar(f"charts/Episodic Return for {n} pedestrians env", score, eps)
                writer.add_scalar(f"charts/Average Linear Velocity per Episode(km/h) for {n} pedestrians env", np.mean(speeds), eps)
                writer.add_scalar(f"charts/Percentage of Covered Distance per Episode for {n} pedestrians env", covered_dist, eps)
                writer.add_scalar(f"charts/Episode Terminated by Collision for {n} pedestrians env", col_with_ped, eps)