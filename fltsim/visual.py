import simplekml
import random

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
