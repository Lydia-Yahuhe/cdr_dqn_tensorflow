"""
1. 检查飞行引擎
2. 检查冲突探测
3. 检查执行动作
"""
import time
import random

import numpy as np
from matplotlib import pyplot as plt

from fltenv import AircraftAgentSet
from fltenv.cmd import int_2_atc_cmd
from fltenv.env import ConflictEnv
from fltsim.load import load_and_split_data

# scenarios, _ = load_and_split_data('scenarios_gail_final', split_ratio=1.0)
scenarios, _ = (1, 1)


# everything is ok!
def scenario_run_to_end():
    for i, scenario in enumerate(scenarios):
        conflict_ac, clock = scenario['conflict_ac'], scenario['time']
        start, end = scenario['start'], scenario['end']
        print(i, conflict_ac, clock, start, end)

        for fpl in scenario['fpl_list']:
            print(fpl.id, fpl.aircraft.id, fpl.routing.id, fpl.min_alt, fpl.max_alt, fpl.startTime)

        agent_set = AircraftAgentSet(fpl_list=scenario['fpl_list'], start=scenario['start'])
        start = time.time()
        while True:
            agent_set.do_step()

            now = agent_set.time
            ac_en = agent_set.agent_en
            if now % 3000 == 0:
                print(now, len(ac_en), time.time() - start)

            if now > end and len(ac_en) <= 0:
                print(now, len(ac_en), time.time() - start)
                break
        agent_set.visual()

        break


# everything is ok!
def run_and_detect():
    for i, scenario in enumerate(scenarios):
        conflict_ac, clock = scenario['conflict_ac'], scenario['time']
        start, end = scenario['start'], scenario['end']
        print(i, conflict_ac, clock, start, end)

        for fpl in scenario['fpl_list']:
            print(fpl.id, fpl.aircraft.id, fpl.routing.id, fpl.min_alt, fpl.max_alt, fpl.startTime)

        agent_set = AircraftAgentSet(fpl_list=scenario['fpl_list'], start=scenario['start'])

        start = time.time()
        while True:
            agent_set.do_step()
            for c in agent_set.detect_conflict_list():
                print(c.to_dict())

            now = agent_set.time
            ac_en = agent_set.agent_en
            if now % 3000 == 0:
                print(now, len(ac_en), time.time() - start)

            if now > end and len(ac_en) <= 0:
                print(now, len(ac_en), time.time() - start)
                break
        agent_set.visual()
        break


def visual_alt_change(tracks):
    x, y = [], []
    for clock, track in tracks.items():
        x.append(clock)
        y.append(track[2])
        print(clock, track[2])
    plt.plot(x, y)
    plt.show()


def run_and_detect_and_resolve():
    for i, scenario in enumerate(scenarios):
        conflict_ac, clock = scenario['conflict_ac'], scenario['time']
        start, end = scenario['start'], scenario['end']
        print(i, conflict_ac, clock, start, end)

        agent_set = AircraftAgentSet(fpl_list=scenario['fpl_list'], start=scenario['start'])
        cmd_list = {'CHB6257': [6349 - 270, 2]}

        start = time.time()
        while True:
            now = agent_set.time
            for key, cmd in cmd_list.items():
                if now == cmd[0] - 1:
                    agent = agent_set.agents[key]
                    cmd = int_2_atc_cmd(cmd[0], cmd[1], agent)
                    agent.assign_cmd(cmd)
                    print(cmd)
                    print(agent.control)

            agent_set.do_step()
            for c in agent_set.detect_conflict_list():
                [a0, a1] = c.id.split('-')
                a0 = agent_set.agents[a0]
                a1 = agent_set.agents[a1]
                if c.time - a0.fpl.startTime < 300 or c.time - a1.fpl.startTime < 300:
                    continue
                print(c.to_dict())

            now = agent_set.time
            ac_en = agent_set.agent_en
            if now % 3000 == 0:
                print(now, len(ac_en), time.time() - start)

            if now > end and len(ac_en) <= 0:
                print(now, len(ac_en), time.time() - start)
                break
        agent_set.visual(save_path='after_resolve')
        # visual_alt_change(agent_set.agents['CHB6257'].tracks)
        break


def run_and_resolve_and_visual():
    env = ConflictEnv()

    size = len(env.test)
    episode = 1
    count = 0
    while not env.test_over():
        print(episode, size)

        obs = env.reset(test=True)
        while True:
            env_action = random.randint(0, 35)
            next_obs, rew, done, result = env.step(env_action)
            obs = next_obs
            if done:
                if result['result']:
                    count += 1
                    print(env.scene.conflict_ac)
                    env.scene.agentSet.do_step(duration=env.scene.clock+300, basic=True)
                    env.scene.agentSet.visual(limit=env.scene.conflict_ac)
                break
        episode += 1
        break

    print('Success Rate is {}%'.format(count * 100.0 / size))


if __name__ == '__main__':
    # scenario_run_to_end()
    # run_and_detect()
    # run_and_detect_and_resolve()
    # run_and_resolve_and_visual()
    pass
