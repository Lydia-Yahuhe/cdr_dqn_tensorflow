from __future__ import annotations
import math
from fltsim.utils import KM2M


# 当到了计划起飞时间时，用fpl初始化profile的参数。
def reset_profile_with_fpl(profile, fpl):
    profile.route = fpl.routing
    profile.make_leg_from_waypoint()

    profile.curLegIdx = 0
    profile.update_cur_next_leg()


def update_profile(profile, status):
    # 判断是否该向下一个航段飞行
    if target_passed(profile, status):
        profile.curLegIdx += 1
        if not profile.update_cur_next_leg():
            status.change_phase('Finished')
            return

    # 根据下一个点更新到下一个点的距离和航向
    target = profile.target
    if not target:
        profile.distToTarget = 0
        profile.courseToTarget = 0
    else:
        profile.distToTarget = status.location.distance_to(target.location)
        profile.courseToTarget = status.location.bearing(target.location)


# 判断是否通过了此Leg的终点
def target_passed(profile, phsyData):
    dist = profile.distToTarget
    if dist >= 20 * KM2M:  # 如果距离大于20KM，判断未经过此航段的end
        return False

    h_spd = phsyData.hSpd
    if dist < h_spd * 1:
        return True

    if profile.nextLeg is not None:
        turnAngle = (profile.nextLeg.course - profile.curLeg.course) % 360
        if dist <= calc_turn_prediction(h_spd, turnAngle, phsyData.performance.normTurnRate):
            return True
    diff = (phsyData.heading - profile.courseToTarget) % 360
    return 270 > diff >= 90


# 计算转弯提前量
def calc_turn_prediction(spd, turnAngle, turnRate):
    if turnAngle > 180:
        turnAngle = turnAngle - 360
    turnAngle = abs(turnAngle)
    turnRadius = spd / math.radians(turnRate)
    return turnRadius * math.tan(math.radians(turnAngle / 2))  # magic
