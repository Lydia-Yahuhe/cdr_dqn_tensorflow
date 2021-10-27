import pymongo
import numpy as np
from tqdm import tqdm

from fltsim.model import Waypoint, Routing, Aircraft, aircraftTypes, FlightPlan, DataSet, Point2D, ConflictScenarioInfo


def load_waypoint(db):
    wpts = {}
    cursor = db['Waypoint'].find()
    for pt in cursor:
        wpt = Waypoint(id=pt['id'], location=Point2D(pt['point']['lng'], pt['point']['lat']))
        wpts[wpt.id] = wpt

    cursor = db["Airport"].find()
    for pt in cursor:
        wpt = Waypoint(id=pt['id'], location=Point2D(pt['location']['lng'], pt['location']['lat']))
        wpts[wpt.id] = wpt

    return wpts


def load_routing(db, wpts):
    ret = {}
    cursor = db['Routing'].find()
    # for e in cursor:
    #     wptList = [wpts[e["departureAirport"]]]
    #     for wptId in e['waypointList']:
    #         wptList.append(wpts[wptId])
    #     wptList.append(wpts[e["arrivalAirport"]])
    #     r = Routing(e['id'], wptList)
    #     ret[r.id] = r

    for e in cursor:
        wptList = []
        for wptId in e['waypointList']:
            wptList.append(wpts[wptId])
        r = Routing(e['id'], wptList)
        ret[r.id] = r

    return ret


def load_aircraft(db):
    ret = {}
    cursor = db['Aircraft'].find()
    for e in cursor:
        info = Aircraft(id=e['id'], aircraftType=aircraftTypes[e['aircraftType']])
        ret[info.id] = info

    return ret


def load_flight_plan(db, aircraft, routes):
    ret = {}
    cursor = db['FlightPlan'].find()
    for e in cursor:
        a = aircraft[e['aircraft']]

        fpl = FlightPlan(
            id=e['id'],
            min_alt=0,
            routing=routes[e['routing']],
            startTime=e['startTime'],
            aircraft=a,
            max_alt=e['flightLevel']
        )

        ret[fpl.id] = fpl

    return ret


def load_data_set():
    connection = pymongo.MongoClient('localhost')
    database = connection['admin']

    wpts = load_waypoint(database)
    aircrafts = load_aircraft(database)
    routes = load_routing(database, wpts)
    fpls = load_flight_plan(database, aircrafts, routes)

    connection.close()
    return DataSet(wpts, routes, fpls, aircrafts)


routings = load_data_set().routings
database = pymongo.MongoClient('localhost')['admin']


def load_and_split_data(path='scenarios_gail', size=None, split_ratio=0.8):
    data = load_data(path)
    if size is None:
        size = len(data)
    else:
        size = min(size, len(data))

    split_size = int(size * split_ratio)
    return data[:split_size], data[split_size:size]


def load_data(collection):
    scenes = []

    data = list(database[collection].find())
    i = 0
    for e in tqdm(data, desc='Loading from ' + collection):
        conflict_ac, clock = e['id'].split('-'), e['time'],
        other = [e['pos0'], e['pos1'], e['hDist'], e['vDist']]

        fpl_list = []
        fpl_list_ac = []
        starts = []
        for f in e['fpl_list']:
            # aircraft
            ac = Aircraft(id=f['aircraft'], aircraftType=aircraftTypes[f['acType']])

            # routing
            r_other = f['other']
            routing = routings[f['routing']]
            wpt_list = routing.waypointList[r_other[0]:r_other[1]]
            routing = Routing(id=routing.id, waypointList=wpt_list, other=r_other)

            # fpl
            startTime = f['startTime']
            starts.append(startTime)
            fpl = FlightPlan(id=f['id'], aircraft=ac, routing=routing, startTime=startTime,
                             min_alt=f['min_alt'], max_alt=f['max_alt'])

            if f['id'] not in conflict_ac:
                fpl_list.append(fpl)
            else:
                fpl_list_ac.append(fpl)

        i += 1
        fpl_list_ac += fpl_list[:2]
        scenes.append(ConflictScenarioInfo(id='No.{}'.format(i), time=clock, conflict_ac=conflict_ac, other=other,
                                           start=min(starts) - 1, end=max(starts), fpl_list=fpl_list_ac))

        if i >= 10000:
            break

    np.random.shuffle(scenes)
    return scenes
