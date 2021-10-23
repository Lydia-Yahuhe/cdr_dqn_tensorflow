from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class FlightPerformance:
    altitude: float = 0
    minClimbTAS: float = 0
    normClimbTAS: float = 0
    maxClimbTAS: float = 0
    climbFuel: float = 0
    minDescentTAS: float = 0
    normDescentTAS: float = 0
    maxDescentTAS: float = 0
    descentFuel: float = 0
    minCruiseTAS: float = 0
    normCruiseTAS: float = 0
    maxCruiseTAS: float = 0
    cruiseFuel: float = 0
    normClimbRate: float = 0
    maxClimbRate: float = 0
    normDescentRate: float = 0
    maxDescentRate: float = 0
    normTurnRate: float = 0
    maxTurnRate: float = 0

    def interpolate(self, r: float, other: FlightPerformance, val: FlightPerformance):
        # pylint: disable=no-member
        for k in self.__dataclass_fields__:
            v1 = getattr(self, k)
            v2 = getattr(other, k)
            v = (v2 - v1) * r + v1
            setattr(val, k, v)
        # keys = self.__dict__.keys()
        # for k in keys:
        #     if k.startswith('__'):
        #         continue
        #     v1 = getattr(self, k)
        #     v2 = getattr(other, k)
        #     v = (v2-v1) * r + v1
        #     setattr(val, k, v)

    def copy(self, other):
        # keys = self.__dict__.keys()
        # for k in keys:
        # if k.startswith('__'):
        #     continue
        # pylint: disable=no-member
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(other, k))

    def set(self, other):
        # keys = self.__dict__.keys()
        # for k in keys:
        # if k.startswith('__'):
        #     continue
        # pylint: disable=no-member
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(other, k))

    def min_max_spd(self, value, v_spd=0.0):
        if v_spd == 0:
            min_limit, max_limit = self.minCruiseTAS, self.maxCruiseTAS
        elif v_spd > 0:
            min_limit, max_limit = self.minClimbTAS, self.maxClimbTAS
        else:
            min_limit, max_limit = self.minDescentTAS, self.minDescentTAS

        return min(max(value, min_limit), max_limit)


@dataclass
class AircraftType:
    id: str
    normAcceleration: float
    maxAcceleration: float
    normDeceleration: float
    maxDeceleration: float
    liftOffSpeed: float
    flightPerformanceTable: List[FlightPerformance]


def compute_performance(acftType: AircraftType, alt: float, val: FlightPerformance):
    table = acftType.flightPerformanceTable
    maxalt = len(table) * 100.0 - 100.0
    if alt > maxalt:
        val.copy(table[len(table) - 1])
        val.altitude = alt
        return
    if alt < table[0].altitude:
        val.copy(table[0])
        val.altitude = alt
        return
    idx = int(alt / 100.0)
    if alt % 100.0 == 0:
        val.copy(table[idx])
        return
    f1 = table[idx]
    f2 = table[idx + 1]
    r = (f2.altitude - alt) / (f2.altitude - f1.altitude)
    f1.interpolate(r, f2, val)


def dictToAircraftType(d):
    d2 = {}
    for k in d:
        if k != 'flightPerformanceTable':
            d2[k] = d[k]
        else:
            table = []
            table0 = d[k]
            for fp in table0:
                table.append(FlightPerformance(**fp))
            d2[k] = table
    return AircraftType(**d2)


def load():
    import os
    p = os.path.abspath(os.path.dirname(__file__))
    p = p + '/AircraftType.json'
    import json
    with open(p, 'r') as f:
        atLst = json.load(f)
    ret = {}
    for at in atLst:
        at1 = dictToAircraftType(at)
        ret[at1.id] = at1
    return ret


aircraftTypes = load()
