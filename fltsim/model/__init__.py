from typing import Dict

from .aircrafttype import *

from fltsim.utils import distance_point2d, bearing_point2d, destination, move_point2d


@dataclass
class Point2D(object):
    lng: float = 0.0
    lat: float = 0.0

    def reset(self, other):
        self.lng = other.lng
        self.lat = other.lat

    def set(self, other):
        self.lng = other.lng
        self.lat = other.lat

    def toArray(self):
        return [self.lng, self.lat]

    def toTuple(self):
        return self.lng, self.lat

    def clear(self):
        self.lng = 0
        self.lat = 0

    def distance_to(self, other):
        return distance_point2d(self, other)

    def bearing(self, other):
        return bearing_point2d(self, other)

    def destination(self, course: float, dist: float):
        coords = destination(self.toTuple(), course, dist)
        return Point2D(lng=coords[0], lat=coords[1])

    def move(self, course: float, dist: float):
        move_point2d(self, course, dist)

    def copy(self):
        return Point2D(lng=self.lng, lat=self.lat)

    def __str__(self):
        return '<%.5f,%.5f>' % (self.lng, self.lat)

    def __repr(self):
        return '<%.5f,%.5f>' % (self.lng, self.lat)


@dataclass
class Waypoint:
    id: str
    location: Point2D

    def __str__(self):
        return '[%s, %s]' % (self.id, self.location)

    def __repr__(self):
        return '[%s, %s]' % (self.id, self.location)

    def distance_to(self, other):
        return self.location.distance_to(other.location)

    def bearing(self, other):
        return self.location.bearing(other.location)

    def copy(self, name='Dogleg'):
        loc = self.location
        return Waypoint(id=name, location=Point2D(loc.lng, loc.lat))


@dataclass
class Aircraft:
    id: str
    aircraftType: AircraftType
    airline: str = None


@dataclass
class Routing:
    id: str
    waypointList: List[Waypoint]
    other: List[int] = None


@dataclass
class FlightPlan:
    id: str
    routing: Routing
    startTime: int
    aircraft: Aircraft
    min_alt: float
    max_alt: float

    def to_dict(self):
        return dict(id=self.id, routing=self.routing.id, startTime=self.startTime,
                    aircraft=self.aircraft.id, acType=self.aircraft.aircraftType.id,
                    min_alt=self.min_alt, max_alt=self.max_alt, other=self.routing.other)


@dataclass
class DataSet:
    waypoints: Dict[str, Waypoint]
    routings: Dict[str, Routing]
    flightPlans: Dict[str, FlightPlan]
    aircrafts: Dict[str, Aircraft]


@dataclass
class ConflictScenarioInfo:
    id: str
    time: int
    conflict_ac: List[str]
    other: List[object]
    start: int
    end: int
    fpl_list: List[FlightPlan]

    def to_dict(self):
        [_, _, h_dist, v_dist] = self.other
        return dict(id=self.id, time=self.time, c_ac=self.conflict_ac,
                    fpl=len(self.fpl_list), h_dist=round(h_dist, 1), v_dist=round(v_dist, 1))
