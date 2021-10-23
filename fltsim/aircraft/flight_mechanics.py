from __future__ import annotations

from fltsim.model import compute_performance


def reset_status_with_fpl(pdata, fpl):
    pdata.alt = fpl.min_alt  # fpl中RFL为航空器的起始高度，alt为目标高度
    compute_performance(pdata.acType, fpl.min_alt, pdata.performance)
    performance = pdata.performance
    pdata.hSpd = performance.normCruiseTAS
    pdata.vSpd = 0

    [wpt0, wpt1] = fpl.routing.waypointList[:2]
    pdata.location.set(wpt0.location)
    pdata.heading = wpt0.bearing(wpt1)
    pdata.change_phase(mode='EnRoute')


def update_status(phsyData, guidance):
    move_horizontal(phsyData, guidance)
    move_vertical(phsyData, guidance)
    update_performance(phsyData)


def move_horizontal(pdata, guidance):
    # 更新水平速度
    preHSpd = pdata.hSpd
    if pdata.hSpd > guidance.targetHSpd:    # 如果当前速度大于目标速度，以正常加速度减速
        dec = pdata.acType.normDeceleration
        pdata.hSpd = max(preHSpd - dec * 1, guidance.targetHSpd)
    elif pdata.hSpd < guidance.targetHSpd:  # 如果当前速度小于目标速度，以正常加速度加速
        acc = pdata.acType.normAcceleration
        pdata.hSpd = min(preHSpd + acc * 1, guidance.targetHSpd)
        
    # 更新航向
    performance = pdata.performance
    diff = (guidance.targetCourse - pdata.heading) % 360
    diff = diff-360 if diff > 180 else diff  # diff ∈ (-180, 180]
    if abs(diff) > 90:  # 如果航向与目标航向相差大于90°，则以最大转弯率转弯
        turn = performance.maxTurnRate * 1
    else:               # 当航向与目标航向相差大于90°时，改为正常转弯率转弯
        turn = performance.normTurnRate * 1
    diff = min(max(-turn, diff), turn)
    pdata.heading = (pdata.heading + diff) % 360

    # 更新位置（经纬度）
    pdata.location.move(pdata.heading, (preHSpd + pdata.hSpd) * 1 / 2)


def move_vertical(pdata, guidance):
    diff = guidance.targetAlt - pdata.alt

    # 以正常上升下降率垂直移动
    if diff < 0:
        v_spd = max(-pdata.performance.normDescentRate * 1, diff)
    elif diff > 0:
        v_spd = min(pdata.performance.normClimbRate * 1, diff)
    else:
        v_spd = 0

    pdata.alt += v_spd
    pdata.vSpd = v_spd


def update_performance(pdata):
    if pdata.performance.altitude != pdata.alt:
        compute_performance(pdata.acType, pdata.alt, pdata.performance)
