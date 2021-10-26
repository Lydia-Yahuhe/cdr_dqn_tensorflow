import simplekml
import random
import numpy as np
import cv2

from fltsim.load import routings
from fltsim.utils import pnpoly, convert_coord_to_pixel, destination, NM2M

alt_mode = simplekml.AltitudeMode.absolute


def make_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = simplekml.Color.rgb(r, g, b, 100)
    return c


def tuple2kml(kml, name, tracks, color=simplekml.Color.chocolate):
    ls = kml.newlinestring(name=name)
    ls.coords = [(wpt[0], wpt[1], wpt[2]) for wpt in tracks]
    ls.extrude = 1
    ls.altitudemode = alt_mode
    ls.style.linestyle.width = 1
    ls.style.polystyle.color = color
    ls.style.linestyle.color = color


def place_mark(point, kml, name='test', hdg=None, description=None):
    pnt = kml.newpoint(name=name, coords=[point],
                       altitudemode=alt_mode, description=description)

    pnt.style.labelstyle.scale = 0.25
    pnt.style.iconstyle.icon.href = '.\\placemark.png'
    # pnt.style.iconstyle.icon.href = '.\\plane.png'
    if hdg is not None:
        pnt.style.iconstyle.heading = (hdg + 270) % 360


def save_to_kml(tracks, plan, save_path='agent_set'):
    kml = simplekml.Kml()

    folder = kml.newfolder(name='real')
    for key, t in tracks.items():
        tuple2kml(folder, key, t, color=simplekml.Color.chocolate)

    folder = kml.newfolder(name='plan')
    for key, t in plan.items():
        tuple2kml(folder, key, t, color=simplekml.Color.royalblue)

    print("Save to "+save_path+".kml successfully!")
    kml.save(save_path+'.kml')


# ---------
# opencv
# ---------
def search_routing_in_a_area(vertices):
    segments = {}
    check_list = []
    for key, routing in routings.items():
        wpt_list = routing.waypointList

        in_poly_idx = []
        for i, wpt in enumerate(wpt_list):
            loc = wpt.location
            in_poly = pnpoly(vertices, [loc.lng, loc.lat])
            if in_poly:
                in_poly_idx.append(i)

        if len(in_poly_idx) <= 0:
            continue

        size = i + 1
        min_idx, max_idx = max(min(in_poly_idx) - 1, 0), min(size, max(in_poly_idx) + 2)
        # print(key, min_idx, max_idx, in_poly_idx, size, len(wpt_list))

        new_wpt_list = wpt_list[min_idx:max_idx]
        assert len(new_wpt_list) >= 2
        for i, wpt in enumerate(new_wpt_list[1:]):
            last_wpt = new_wpt_list[i]
            name_f, name_l = last_wpt.id + '-' + wpt.id, wpt.id + '-' + last_wpt.id

            if name_f not in check_list:
                segments[name_f] = [[last_wpt.location.lng, last_wpt.location.lat],
                                    [wpt.location.lng, wpt.location.lat]]
                check_list += [name_l, name_f]

    return segments


def generate_wuhan_base_map(size=(700, 1000, 3), save=None, show=False, **kwargs):
    # 武汉空域
    vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
                (114.07, 32.125), (115.81333333333333, 32.90833333333333),
                (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
                (113.12, 29.383333333333333), (109.4, 29.516666666666666),
                (109.51666666666667, 31.9), (109.51666666666667, 31.9)]

    # 创建一个宽512高512的黑色画布，RGB(0,0,0)即黑色
    image = np.zeros(size, np.uint8)

    points = convert_coord_to_pixel(vertices, **kwargs)

    segments = search_routing_in_a_area(vertices)
    for name, coord in segments.items():
        coord_idx = convert_coord_to_pixel(coord, **kwargs)
        cv2.line(image, coord_idx[0], coord_idx[1], (205, 205, 193), 1)
    pts = np.array(points, np.int32).reshape((-1, 1, 2,))
    cv2.polylines(image, [pts], True, (255, 191, 0), 2)

    if save is not None:
        cv2.imwrite(save, image)

    if show:
        cv2.imshow("wuhan", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    return image


# 点的颜色为
def add_points_on_base_map(points, image, save=False, display=False, font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX,
                           **kwargs):
    radius = 10
    points_just_coord = []
    for [name, is_c_ac, lng, lat, alt, *point] in points:
        coord = [lng, lat]
        coord_idx = convert_coord_to_pixel([coord], **kwargs)[0]

        blue = min(255, max((alt-6000) / 6000 * 255, 0))
        if is_c_ac:
            cv2.circle(image, coord_idx, radius, (0, 0, blue), -1)
        else:
            cv2.circle(image, coord_idx, radius, (0, 255-blue, 255), -1)

        heading_spd_point = destination(coord, point[2], 180/3600*point[0]*NM2M)
        add_lines_on_base_map([[coord, heading_spd_point, False]], image, display=False)

        if display and is_c_ac:
            [x, y] = coord_idx
            text_color = (255, 255, 255)  # BGR
            decimal = 1

            cv2.putText(image, name, (x, y+10), font, font_scale, text_color, 1)
            state = 'Altitude: {}'.format(round(alt, decimal))
            cv2.putText(image, state, (x, y+30), font, font_scale, text_color, 1)
            state = '   Speed: {}({})'.format(round(point[0], decimal), round(point[1], decimal))
            cv2.putText(image, state, (x, y+50), font, font_scale, text_color, 1)
            state = ' Heading: {}'.format(round(point[2], decimal))
            cv2.putText(image, state, (x, y+70), font, font_scale, text_color, 1)

        points_just_coord.append((lng, lat, alt))

    if save:
        cv2.imwrite("script/wuhan.jpg", image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, points_just_coord


def add_texts_on_base_map(texts, image, start=(750, 70), color=(255, 255, 255), save=False,
                          font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX):
    for i, (key, text) in enumerate(texts.items()):
        string = key + ': {}'.format(text)
        cv2.putText(image, string, (start[0], start[1]+i*20), font, font_scale, color, 1)

    if save:
        cv2.imwrite("script/wuhan.jpg", image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def add_lines_on_base_map(lines, image, save=False, color=(154, 250, 0), display=True, font_scale=0.4,
                          font=cv2.FONT_HERSHEY_SIMPLEX, **kwargs):
    if len(lines) <= 0:
        return image

    decimal = 1
    for [pos0, pos1, *other] in lines:
        if other[-1]:
            color = (255, 130, 171)

        [start, end] = convert_coord_to_pixel([pos0, pos1], **kwargs)
        cv2.line(image, start, end, color, 1)

        if display:
            [h_dist, v_dist] = other[:2]
            mid_idx = (int((start[0]+end[0])/2)+10, int((start[1]+end[1])/2)+10)
            state = ' H_dist: {}, V_dist: {}'.format(round(h_dist, decimal), round(v_dist, decimal))
            cv2.putText(image, state, mid_idx, font, font_scale, color, 1)

    if save:
        cv2.imwrite("script/wuhan.jpg", image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


# kwargs = dict(border=[108, 118, 28, 35], scale=100)
# generate_wuhan_base_map(save='wuhan_base.jpg', show=True, **kwargs)
