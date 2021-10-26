from __future__ import annotations
from dataclasses import dataclass, field

from typing import List
from enum import Enum

from fltsim.model import Routing, Waypoint, aircraftTypes, AircraftType, FlightPerformance, Point2D
from fltsim.aircraft import atccmd

FPLPhase = Enum('FPLPhase', ('Schedule', 'EnRoute', 'Finished'))


@dataclass
class FlightControl:
    altCmd: atccmd.AltCmd = None
    spdCmd: atccmd.SpdCmd = None
    hdgCmd: atccmd.HdgCmd = None

    def set(self, other: FlightControl):
        self.altCmd = other.altCmd
        self.spdCmd = other.spdCmd
        self.hdgCmd = other.hdgCmd

    def transition(self, mode='Alt'):
        if mode == 'Alt':
            self.altCmd = None
        elif mode == 'Spd':
            self.spdCmd = None
        elif mode == 'Hdg':
            self.hdgCmd = None
        else:
            raise NotImplementedError

    def to_string(self):
        return self.altCmd, self.spdCmd, self.hdgCmd


@dataclass
class FlightGuidance:
    targetAlt: float = 0
    targetHSpd: float = 0
    targetCourse: float = 0

    def set(self, other: FlightGuidance):
        self.targetAlt = other.targetAlt
        self.targetHSpd = other.targetHSpd
        self.targetCourse = other.targetCourse


@dataclass
class FlightLeg:
    start: Waypoint
    end: Waypoint
    distance: float = 0
    course: float = 0

    def __post_init__(self):
        self.distance = self.start.distance_to(self.end)
        self.course = self.start.bearing(self.end)

    def copy(self) -> FlightLeg:
        return FlightLeg(self.start, self.end)


@dataclass
class FlightProfile(object):
    route: Routing = None
    legs: List[FlightLeg] = None
    curLegIdx: int = 0
    curLeg: FlightLeg = None
    nextLeg: FlightLeg = None
    distToTarget: float = 0
    courseToTarget: float = 0

    @property
    def target(self):
        if not self.curLeg:
            return None
        return self.curLeg.end

    def update_cur_next_leg(self):
        cur_idx = self.curLegIdx
        legs_size = len(self.legs) - 1  # 若总共9个航段，则cueLeg ∈ [0,8]
        self.curLeg = self.legs[cur_idx] if cur_idx <= legs_size else None
        self.nextLeg = self.legs[cur_idx+1] if cur_idx+1 <= legs_size else None

        # 如果curLeg和nextLeg都是None，则飞行计划结束
        if self.curLeg is None and self.nextLeg is None:
            return False

        self.distToTarget = self.curLeg.distance
        self.courseToTarget = self.curLeg.course
        return True

    def next(self, delta=1):
        start_idx = self.curLegIdx
        wptList = self.route.waypointList
        return wptList[start_idx+1: min(start_idx+1+delta, len(wptList)-1)]

    def set(self, other: FlightProfile):
        self.route = other.route

        self.legs = None if other.legs is None else other.legs[:]
        self.curLegIdx = other.curLegIdx
        self.curLeg = other.curLeg
        self.nextLeg = other.nextLeg
        self.distToTarget = other.distToTarget
        self.courseToTarget = other.courseToTarget

    def make_leg_from_waypoint(self, wpt_list=None):
        if wpt_list is None:
            wpt_list = self.route.waypointList

        ret = []
        for i, point in enumerate(wpt_list[1:]):
            ret.append(FlightLeg(wpt_list[i], point))
        self.legs = ret


@dataclass
class FlightStatus:
    hSpd: float = 0
    vSpd: float = 0
    alt: float = 0
    heading: float = 0
    acType: AircraftType = aircraftTypes['A320']
    location: Point2D = field(default_factory=Point2D)
    performance: FlightPerformance = field(default_factory=FlightPerformance)
    phase: FPLPhase = FPLPhase.Schedule

    @property
    def course(self):
        return self.heading

    def change_phase(self, mode='EnRoute'):
        if mode == 'EnRoute':
            self.phase = FPLPhase.EnRoute
        elif mode == 'Finished':
            self.phase = FPLPhase.Finished
        elif mode == 'Schedule':
            self.phase = FPLPhase.Schedule
        else:
            raise NotImplementedError

    def is_enroute(self):
        return self.phase == FPLPhase.EnRoute

    def is_finished(self):
        return self.phase == FPLPhase.Finished

    def set(self, other: FlightStatus):
        self.hSpd = other.hSpd
        self.vSpd = other.vSpd
        # pylint: disable=no-member
        self.location.reset(other.location)
        self.alt = other.alt
        self.heading = other.heading
        self.acType = other.acType
        # pylint: disable=no-member
        self.performance.copy(other.performance)
        self.phase = other.phase

    def x_data(self):
        return self.location.toArray()+[self.alt, self.hSpd, self.vSpd, self.heading]
