from math import *
from rtree import index
import math

R = 6371393.0  # m
DEG_RADIAN = pi / 180.0
RADIAN_DEG = 180.0 / pi
G0 = 0.982
epsilon = 7.0 / 3 - 4.0 / 3 - 1.0
KM2M = 1000.0
M2KM = 1.0 / KM2M
KT2MPS = 0.514444444444444
NM2M = 1852


def distance_point2d(d0, d1):
    lng0 = radians(d0.lng)
    lat0 = radians(d0.lat)
    lng1 = radians(d1.lng)
    lat1 = radians(d1.lat)
    dlng = lng0 - lng1
    dlat = lat0 - lat1
    tmp1 = sin(dlat / 2)
    tmp2 = sin(dlng / 2)
    a = tmp1 * tmp1 + cos(lat0) * cos(lat1) * tmp2 * tmp2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def distance(c0, c1):
    lng0 = c0[0] * DEG_RADIAN
    lat0 = c0[1] * DEG_RADIAN
    lng1 = c1[0] * DEG_RADIAN
    lat1 = c1[1] * DEG_RADIAN
    dlng = lng0 - lng1
    dlat = lat0 - lat1
    tmp1 = sin(dlat / 2)
    tmp2 = sin(dlng / 2)
    a = tmp1 * tmp1 + cos(lat0) * cos(lat1) * tmp2 * tmp2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def bearing_point2d(d0, d1):
    lng0 = radians(d0.lng)
    lat0 = radians(d0.lat)
    lng1 = radians(d1.lng)
    lat1 = radians(d1.lat)
    dlng = lng1 - lng0
    coslat1 = cos(lat1)
    tmp1 = sin(dlng) * coslat1
    tmp2 = cos(lat0) * sin(lat1) - sin(lat0) * coslat1 * cos(dlng)
    return (atan2(tmp1, tmp2) * RADIAN_DEG) % 360


def bearing(c0, c1):
    lng0 = c0[0] * DEG_RADIAN
    lat0 = c0[1] * DEG_RADIAN
    lng1 = c1[0] * DEG_RADIAN
    lat1 = c1[1] * DEG_RADIAN
    dlng = lng1 - lng0
    coslat1 = cos(lat1)
    tmp1 = sin(dlng) * coslat1
    tmp2 = cos(lat0) * sin(lat1) - sin(lat0) * coslat1 * cos(dlng)
    return (atan2(tmp1, tmp2) * RADIAN_DEG) % 360


def intersection(a0, a1):
    c = (a0 - a1) % 360
    if c > 180:
        return c - 360
    return c


def move_point2d(src, course, dist):
    lng1 = radians(src.lng)
    lat1 = radians(src.lat)
    r = dist / R
    course = radians(course)
    cosR = cos(r)
    sinR = sin(r)
    sinLat1 = sin(lat1)
    cosLat1 = cos(lat1)

    lat2 = asin(sinLat1 * cosR + cosLat1 * sinR * cos(course))
    lng2 = lng1 + atan2(sin(course) * sinR * cosLat1, cosR - sinLat1 * sin(lat2))
    src.lng = degrees(lng2)
    src.lat = degrees(lat2)


def get_third(former, heading, dist):
    heading = (360 + heading) % 360
    later = former.copy()
    move_point2d(later.location, heading, dist)
    return later


def destination(src, course: float, dist: float):
    lng1 = src[0] * DEG_RADIAN
    lat1 = src[1] * DEG_RADIAN
    r = dist / R
    course = course * DEG_RADIAN
    cosR = cos(r)
    sinR = sin(r)
    sinLat1 = sin(lat1)
    cosLat1 = cos(lat1)

    lat2 = asin(sinLat1 * cosR + cosLat1 * sinR * cos(course))
    lng2 = lng1 + atan2(sin(course) * sinR * cosLat1, cosR - sinLat1 * sin(lat2))
    return lng2 * RADIAN_DEG, lat2 * RADIAN_DEG


def make_bbox(pos, ext=(0, 0, 0)):
    return (pos[0] - ext[0], pos[1] - ext[1], pos[2] - ext[2],
            pos[0] + ext[0], pos[1] + ext[1], pos[2] + ext[2])


def position_in_bbox(bbox, p, delta=(0.05, 0.05, 300)):
    x = int((p[0] - bbox[0]) / delta[0])
    y = int((p[1] - bbox[1]) / delta[1])
    z = int((p[2] - bbox[2]) / delta[2])

    x_size = int(bbox[3] - bbox[0]) / delta[1]
    y_size = int(bbox[4] - bbox[1]) / delta[1]
    z_size = int(bbox[5] - bbox[2]) / delta[2]
    # if x not in range(40) or y not in range(40) or z not in range(10):
    #     return -1

    return x + y * x_size + z * x_size * y_size


def build_rt_index(agents):
    p = index.Property()
    p.dimension = 3
    idx = index.Index(properties=p)
    for i, a in enumerate(agents):
        idx.insert(i, make_bbox(a.position))
    return idx


def build_rt_index_with_list(points):
    p = index.Property()
    p.dimension = 3
    idx = index.Index(properties=p)
    for i, point in enumerate(points):
        idx.insert(i, make_bbox(point))
    return idx


def mid_position(pos0, pos1):
    return (
        (pos0[0] - pos1[0]) / 2 + pos1[0],
        (pos0[1] - pos1[1]) / 2 + pos1[1],
        (pos0[2] - pos1[2]) / 2 + pos1[2])


def format_time(t):
    return '%02d:%02d:%02d:%02d' % (int(t / 24 / 3600), int(t / 3600) % 24, int(t / 60) % 60, int(t) % 60)


def equal(list_a, list_b):
    if len(list_a) != len(list_b):
        return False

    return sum([a not in list_b for a in list_a]) <= 0


def center(locations):
    x, y, z, length = 0, 0, 0, len(locations)
    sum_alt = 0
    for lon, lat, alt in locations:
        lon = radians(float(lon))
        lat = radians(float(lat))

        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)
        sum_alt += alt

    x = float(x / length)
    y = float(y / length)
    z = float(z / length)
    alt = float(sum_alt / length)

    return degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))), alt


# 已知三个点的坐标，计算其三角形面积
def area(a, b, c):
    ab = distance(a, b)
    ac = distance(a, c)
    bc = distance(b, c)
    p = (ab + ac + bc) / 2
    S = math.sqrt(abs(p * (p - ab) * (p - ac) * (p - bc)))

    return S


# 已知三个点的坐标，计算a点到bc的距离
def high(a, b, c):
    S = area(a, b, c)
    return 2 * S / distance(b, c)


flight_level = [i * 300.0 for i in range(29)]
flight_level += [i * 300.0 + 200.0 for i in range(29, 50)]


# 修正高度到高度层, 8300 → 8400
def calc_level(alt, v_spd, delta):
    delta = int(delta / 300.0)
    lvl = int(alt / 300.0) * 300.0

    if alt < 8700.0:
        idx = flight_level.index(lvl)
        if (v_spd > 0 and alt - lvl != 0) or (v_spd == 0 and alt - lvl > 150):
            idx += 1

        return flight_level[idx + delta]

    lvl += 200.0
    idx = flight_level.index(lvl)
    if v_spd > 0 and alt - lvl > 0:
        idx += 1
    elif v_spd < 0 and alt - lvl < 0:
        idx -= 1

    return flight_level[idx + delta]


# 判断一个点是否在一个不规则多边形内
def pnpoly(vertices, testp):
    """
    vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
            (114.07, 32.125), (115.81333333333333, 32.90833333333333),
            (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
            (113.12, 29.383333333333333), (109.4, 29.516666666666666),
            (109.51666666666667, 31.9), (109.51666666666667, 31.9)]
    import simplekml

    kml = simplekml.Kml()

    line = kml.newlinestring(name='sector')
    line.coords = [(wpt[0], wpt[1], 8100.0) for wpt in vertices]
    line.extrude = 1
    line.altitudemode = simplekml.AltitudeMode.absolute
    line.style.linestyle.width = 1

    folder = kml.newfolder(name='points')
    for i in range(1000):
        lng = np.random.randint(0, 8000) / 1000.0+109.0
        lat = np.random.randint(0, 4000) / 1000.0+29.0
        in_poly = pnpoly(vertices, [lng, lat])
        print(i, lng, lat, in_poly)
        if in_poly:
            pnt = folder.newpoint(name=str(i), coords=[(lng, lat, 8100.0)],
                                  altitudemode=simplekml.AltitudeMode.absolute)
    kml.save('test.kml')
    """
    n = len(vertices)
    j = n - 1
    res = False
    for i in range(n):
        if (vertices[i][1] > testp[1]) != (vertices[j][1] > testp[1]) and \
                testp[0] < (vertices[j][0] - vertices[i][0]) * (testp[1] - vertices[i][1]) / (
                vertices[j][1] - vertices[i][1]) + vertices[i][0]:
            res = not res
        j = i
    return res


# n为待转换的十进制数，x为进制，取值为2-16
def convert_with_align(n, x=3, align=4):
    def convert(digit, radix):
        if radix > digit >= 0:
            return digit
        else:
            return convert(digit // radix, radix) * 10 + digit % radix  # 需要细细思考里面的规律

    str_origin = str(convert(n, x))
    return '0'*(align-len(str_origin))+str_origin


def convert_coord_to_pixel(objects, border=(108, 118, 28, 35), scale=100):
    min_x, max_x, min_y, max_y = border
    scale_x = (max_x - min_x) * scale
    scale_y = (max_y - min_y) * scale

    tmp = []
    for [x, y, *_] in objects:
        x_idx = int((x - min_x) / (max_x - min_x) * scale_x)
        y_idx = int((max_y - y) / (max_y - min_y) * scale_y)
        tmp.append([x_idx, y_idx])
    return tmp


def convert_km_to_pixel_number(km, scale=100):
    degree = km / 111
    return int(degree*scale)
