import numpy as np
import time
from contextlib import contextmanager

from baselines.common import colorize

from fltenv.agent_Set import AircraftAgentSet
from fltenv.cmd import CmdCount, int_2_atc_cmd, check_cmd

from fltsim.utils import make_bbox, mid_position, build_rt_index


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end=' ')
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


class ConflictScene:
    def __init__(self, info, limit=0):
        self.conflict_ac, self.clock = info['conflict_ac'], info['time']
        fpl_list = info['fpl_list']

        self.agentSet = AircraftAgentSet(fpl_list=fpl_list, start=info['start'])
        self.agentSet.do_step(self.clock - 300 + limit, basic=True)
        self.conflict_pos = info['other'][0]

        # print('\nNew scenario--------------------------------')
        # print(' Conflict Info: ', self.conflict_ac, self.clock, self.agentSet.time, len(fpl_list), info['other'])

        self.cmd_list = {}
        for c_ac in self.conflict_ac:
            self.cmd_list[c_ac] = []

    def now(self):
        return self.agentSet.time

    def get_states(self):
        state = [[0.0 for _ in range(7)] for _ in range(50)]

        j = 0
        # ghost = AircraftAgentSet(other=self.agentSet)
        # ghost.do_step(self.now() + 60, basic=True)
        # ghost_2 = AircraftAgentSet(other=ghost)
        # ghost_2.do_step(self.now()+120, basic=True)
        for agent in self.agentSet.agent_en:
            pos = agent.position
            ele = [int(agent.id in self.conflict_ac),
                   pos[0] - self.conflict_pos[0],
                   pos[1] - self.conflict_pos[1],
                   (pos[2] - self.conflict_pos[2]) / 3000,
                   (agent.status.hSpd - 150) / 100,
                   agent.status.vSpd / 20,
                   agent.status.heading / 180]

            # agent_g = ghost.agents[agent.id]
            # pos_g = agent_g.position
            # if agent_g.is_enroute():
            #     ele += [pos_g[0] - self.conflict_pos[0],
            #             pos_g[1] - self.conflict_pos[1],
            #             (pos_g[2] - self.conflict_pos[2]) / 3000,
            #             (agent_g.status.hSpd - 150) / 100,
            #             agent_g.status.vSpd / 20,
            #             agent_g.status.heading / 180]
            # else:
            #     ele += [0.0 for _ in range(6)]
            #
            # agent_g = ghost_2.agents[agent.id]
            # pos_g = agent_g.position
            # if agent_g.is_enroute():
            #     ele += [pos_g[0] - self.conflict_pos[0],
            #             pos_g[1] - self.conflict_pos[1],
            #             (pos_g[2] - self.conflict_pos[2]) / 3000,
            #             (agent_g.status.hSpd - 150) / 100,
            #             agent_g.status.vSpd / 20,
            #             agent_g.status.heading / 180]
            # else:
            #     ele += [0.0 for _ in range(6)]

            j = min(50 - 1, j)
            state[j] = ele
            j += 1

        return np.concatenate(state)

    def do_step(self, action):
        agent, idx = self.conflict_ac[action // CmdCount], action % CmdCount
        check = self.cmd_list[agent]

        agent = self.agentSet.agents[agent]
        [hold, *cmd_list] = int_2_atc_cmd(self.now() + 1, idx, agent)
        print(action, hold, end=' ')
        # print(action, hold, cmd, end=' ')

        self.agentSet.do_step(duration=hold)
        conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
        # solved, done, cmd
        if len(conflicts) > 0:
            return False, True, None

        ok_list = []
        for cmd in cmd_list:
            ok, reason = check_cmd(cmd, agent, check)
            ok_list.append(ok)
            if ok:
                agent.assign_cmd(cmd)
        cmd_info = {'cmd': cmd_list, 'ok': ok_list}
        # print(agent.control)
        if self.__do_real(self.now() + 120):
            return False, True, cmd_info

        has_conflict = self.__do_fake(self.clock + 300)
        done = not has_conflict or self.now() - self.clock >= 300
        return not has_conflict, done, cmd_info

    def __do_real(self, end_time):
        while self.now() < end_time:
            self.agentSet.do_step(duration=30)
            conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return True
        return False

    def __do_fake(self, end_time):
        ghost = AircraftAgentSet(other=self.agentSet)
        while ghost.time < end_time:
            ghost.do_step(duration=30)
            conflicts = ghost.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return True
        return False
