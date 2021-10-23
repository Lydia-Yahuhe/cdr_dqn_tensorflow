import time

import numpy as np

import pymongo
import simplekml
from matplotlib import pyplot as plt

from fltenv import AircraftAgentSet, ConflictScene
from fltsim.load import load_data_set, load_and_split_data
from fltsim.model import Routing, FlightPlan
from fltsim.utils import pnpoly

"""
 1. 找到所有经过武汉扇区（vertices）的航路 → wh_routing_list；
 2. 截取wh_routing_list航路中在武汉扇区（vertices）里的航段；
 3. 随机抽取120个routing，构建飞行计划和AgentSet；
 4. 运行AgentSet，并进行冲突探测；
 5. 剔除冲突时间-起飞时间<=600的飞行计划，并重建AgentSet；
 6. 运行AgentSet，并进行冲突探测；
 7. 记录冲突信息和飞行计划 → meta_scenarios；
 8. 记录各个冲突信息和飞行计划 → scenarios_gail；
 9. 复现scenarios_gail中的场景，对冲突航空器进行冲突探测，剔除影响冲突航空器的其它航空器；
10. 将新的场景写入新的数据库 → scenarios_gail_final。
"""
database = pymongo.MongoClient('localhost')['admin']

vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
            (114.07, 32.125), (115.81333333333333, 32.90833333333333),
            (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
            (113.12, 29.383333333333333), (109.4, 29.516666666666666),
            (109.51666666666667, 31.9), (109.51666666666667, 31.9)]


data_set = load_data_set()
ro_set = list(data_set.routings.values())
fpl_set = list(data_set.flightPlans.values())
flight_level = [i * 300.0 for i in range(21, 29)]
flight_level += [i * 300.0 + 200.0 for i in range(29, 41)]
flight_level = {i: level for i, level in enumerate(flight_level)}


def search_routing_in_wuhan(visual=False):
    route_list = []
    for routing in ro_set:
        in_poly_idx = []
        for i, wpt in enumerate(routing.waypointList):
            loc = wpt.location
            in_poly = pnpoly(vertices, [loc.lng, loc.lat])
            if in_poly:
                in_poly_idx.append(i)

        if len(in_poly_idx) > 0:
            # print(min(in_poly_idx) == 0, len(in_poly_idx), max(in_poly_idx) == i)
            route_list.append([routing, in_poly_idx, i+1])

    if visual:
        kml = simplekml.Kml()
        line = kml.newlinestring(name='sector')
        line.coords = [(wpt[0], wpt[1], 8100.0) for wpt in vertices]
        line.extrude = 1
        line.altitudemode = simplekml.AltitudeMode.absolute
        line.style.linestyle.width = 1

        for [route, *_] in route_list:
            points = []
            for wpt in route.waypointList:
                loc = wpt.location
                points.append([loc.lng, loc.lat, 8100.0])

            ls = kml.newlinestring(name=route.id)
            ls.coords = points
            ls.extrude = 1
            ls.altitudemode = simplekml.AltitudeMode.absolute
            ls.style.linestyle.width = 1
        kml.save('test.kml')

    return route_list


def get_fpl_random(routes, number=30):
    np.random.shuffle(routes)
    np.random.shuffle(fpl_set)

    fpl_list, starts = [], []
    for i, [route, idx, size] in enumerate(routes[:number]):
        min_idx, max_idx = max(min(idx)-1, 0), min(size, max(idx)+2)
        wpt_list = route.waypointList[min_idx:max_idx]

        # routing
        routing = Routing(id=route.id, waypointList=wpt_list, other=[min_idx, max_idx])

        # start time
        start_time = np.random.randint(0, 2400)
        starts.append(start_time)

        # min_alt, max_alt
        min_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]

        if np.random.randint(0, 60) % 3 == 0:
            max_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
        else:
            max_alt = min_alt

        # aircraft
        fpl = fpl_set[i]
        ac = fpl.aircraft

        # flight plan
        fpl = FlightPlan(id=fpl.id, routing=routing, aircraft=ac, startTime=start_time,
                         min_alt=min_alt, max_alt=max_alt)
        print(i, fpl.id, ac.id, start_time, routing.id, min_alt, max_alt)
        fpl_list.append(fpl)

    return fpl_list, starts


def agent_set_run(fpl_list, starts, visual=None):
    start, end = min(starts) - 1, max(starts)
    print('>>>', len(fpl_list), start, end, '\n')

    agent_set = AircraftAgentSet(fpl_list=fpl_list, start=start)

    conflicts_all = []
    start = time.time()
    while True:
        agent_set.do_step()
        conflicts = []
        for c in agent_set.detect_conflict_list():
            [a0, a1] = c.id.split('-')
            fpl0 = agent_set.agents[a0].fpl
            fpl1 = agent_set.agents[a1].fpl

            print('-------------------------------------')
            print('|  Conflict ID: ', c.id)
            print('|Conflict Time: ', c.time)
            print('|   H Distance: ', c.hDist)
            print('|   V Distance: ', c.vDist)
            print('|     a0 state: ', c.pos0)
            print('|      a0 info: ', fpl0.startTime, fpl0.min_alt, fpl0.max_alt)
            print('|     a1 state: ', c.pos1)
            print('|      a1 info: ', fpl1.startTime, fpl1.min_alt, fpl1.max_alt)
            print('-------------------------------------\n')
            conflicts.append([c, fpl0, fpl1])
        conflicts_all += conflicts

        now = agent_set.time
        ac_en = agent_set.agent_en
        if now % 600 == 0:
            print(now, len(ac_en), len(conflicts), time.time() - start)

        if now > end and len(ac_en) <= 0:
            print(now, len(ac_en), time.time() - start)
            break

    if visual is not None:
        agent_set.visual(save_path=visual)
    return conflicts_all


def write_in_db(name, conflict_info, fpl_info):
    collection = database['meta_scenarios']
    conflict_list = [c.to_dict() for [c, *_] in conflict_info]
    fpl_list = [fpl.to_dict() for fpl in fpl_info]
    collection.insert(dict(id=name, conflict_list=conflict_list, fpl_list=fpl_list))

    collection = database['scenarios_gail']
    for c_dict in conflict_list:
        c_dict['fpl_list'] = fpl_list
        collection.insert(c_dict)


def main_step_1_to_8():
    route_list = search_routing_in_wuhan()

    # np.random.seed(1234)

    for i in range(1000, 3000):
        fpl_list, starts = get_fpl_random(route_list[:], number=120)
        conflicts = agent_set_run(fpl_list, starts)

        # 去除没有时间解脱的冲突
        shift_list = []
        for [c, fpl0, fpl1] in conflicts:
            [a0, a1] = c.id.split('-')
            if c.time - fpl0.startTime < 600:
                shift_list.append(a0)

            if c.time - fpl1.startTime < 600:
                shift_list.append(a1)

        new_fpl_list, new_starts = [], []
        for fpl in fpl_list:
            if fpl.id not in shift_list:
                new_fpl_list.append(fpl)
                new_starts.append(fpl.startTime)

        conflicts = agent_set_run(new_fpl_list, new_starts)
        new_conflicts = []
        for c in conflicts:
            if c[0].time >= 3000:
                continue
            new_conflicts.append(c)
        write_in_db(i, new_conflicts, new_fpl_list)

        # break


def main_step_9_10():
    info, _ = load_and_split_data(split_ratio=1.0)
    size = len(info)
    collection = database['scenarios_gail_final']

    for i, e in enumerate(info):
        scenario = ConflictScene(e)
        agent_set = scenario.agentSet
        conflict_ac = scenario.conflict_ac

        conflicts = []
        while agent_set.time < scenario.clock + 360:
            agent_set.do_step()
            conflicts += agent_set.detect_conflict_list(search=conflict_ac)

        # assert len(conflicts) == 1
        # print(i, conflict_ac[0] in conflicts[0].id, conflict_ac[1] in conflicts[0].id)

        shift_list = []
        if len(conflicts) == 1:
            main_c = conflicts[0]
        else:
            main_c = None
            for c in conflicts:
                [a0, a1] = c.id.split('-')
                if a0 not in conflict_ac:
                    shift_list.append(a0)
                    continue

                if a1 not in conflict_ac:
                    shift_list.append(a1)
                    continue
                main_c = c

        assert main_c is not None

        print(i, size, len(conflicts), conflict_ac[0] in main_c.id,
              conflict_ac[1] in main_c.id, len(e['fpl_list']), end='\t')
        c_dict = main_c.to_dict()
        fpl_list = [fpl.to_dict() for fpl in e['fpl_list'] if fpl.id not in shift_list]
        print(len(fpl_list))
        c_dict['fpl_list'] = fpl_list
        collection.insert(c_dict)

        # if i >= 99:
        #     break


def check_single_scenario():
    info, _ = load_and_split_data('scenarios_gail_final', split_ratio=1.0)

    for i, e in enumerate(info):
        scenario = ConflictScene(e)
        agent_set = scenario.agentSet
        conflict_ac = scenario.conflict_ac

        conflicts = []
        while agent_set.time < scenario.clock + 360:
            agent_set.do_step()
            conflicts += agent_set.detect_conflict_list(search=conflict_ac)

        assert len(conflicts) == 1
        print(conflicts[0].to_dict())
        print(i, conflict_ac[0] in conflicts[0].id, conflict_ac[1] in conflicts[0].id)


if __name__ == '__main__':
    # main_step_1_to_8()
    # main_step_9_10()
    check_single_scenario()
    pass
