from __future__ import annotations
from dataclasses import dataclass

from fltsim.aircraft import AircraftAgent
from fltsim.utils import make_bbox, distance, build_rt_index
from fltsim.visual import save_to_kml


@dataclass
class Conflict:
    id: str
    time: int
    hDist: float
    vDist: float
    pos0: tuple
    pos1: tuple

    def to_dict(self):
        return dict(id=self.id, time=self.time, hDist=self.hDist, vDist=self.vDist,
                    pos0=self.pos0, pos1=self.pos1)


class AircraftAgentSet:
    def __init__(self, fpl_list=None, start=None, other=None):
        if fpl_list:
            self.time = start or -1
            self.agents = {fpl.id: AircraftAgent(fpl) for fpl in fpl_list}
            self.check_list = []
        else:
            self.time = other.time
            self.agents = {a_id: agent.copy() for a_id, agent in other.agents.items()}
            self.check_list = other.check_list[:]

        self.agent_en = []

    def do_step(self, duration=1, basic=False):
        now = self.time
        duration -= now * int(basic)
        self.agent_en = []

        for key, agent in self.agents.items():
            if agent.is_finished():
                continue

            agent.do_step(now, duration)

            if agent.is_enroute():
                self.agent_en.append(agent)

        self.time = now + duration

    def detect_conflict_list(self, search=None):
        conflicts, agent_en = [], self.agent_en

        if len(agent_en) <= 1:
            return []

        r_tree = build_rt_index(agent_en)
        check_list = []
        for a0 in search or agent_en:
            a0 = self.agents[a0] if isinstance(a0, str) else a0
            bbox = make_bbox(a0.position, (0.1, 0.1, 299))

            for i in r_tree.intersection(bbox):
                a1 = agent_en[i]
                if a0 == a1 or a0.id+'-'+a1.id in check_list+self.check_list:
                    continue

                check_list.append(a1.id+'-'+a0.id)
                self.detect_conflict(a0, a1, conflicts)

        return conflicts

    def detect_conflict(self, a0, a1, conflicts):
        pos0, pos1 = a0.position, a1.position

        h_dist = distance(pos0, pos1)
        v_dist = abs(pos0[2] - pos1[2])
        if h_dist >= 10000 or v_dist >= 300.0:
            return

        self.check_list.append(a0.id + '-' + a1.id)
        self.check_list.append(a1.id + '-' + a0.id)

        conflicts.append(Conflict(id=a0.id+"-"+a1.id, time=self.time, hDist=h_dist, vDist=v_dist, pos0=pos0, pos1=pos1))

    def visual(self, save_path='agentSet', limit=None):
        tracks_real = {}
        tracks_plan = {}
        for a_id, agent in self.agents.items():
            if limit is not None and a_id not in limit:
                continue

            tracks_real[a_id] = [tuple(track[:3]) for track in agent.tracks.values()]
            tracks_plan[a_id] = [(point.location.lng, point.location.lat, 8100.0)
                                 for point in agent.fpl.routing.waypointList]
        save_to_kml(tracks_real, tracks_plan, save_path=save_path)
