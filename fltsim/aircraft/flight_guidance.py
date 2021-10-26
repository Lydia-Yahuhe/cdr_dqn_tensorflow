from __future__ import annotations


def reset_guidance_with_fpl(guidance, fpl):
    guidance.targetAlt = fpl.max_alt
    guidance.targetHSpd = 0
    guidance.targetCourse = 0


# 速度动作被抛弃了
def update_guidance(now, guidance, status, control, profile):
    # 速度引导(均为标称速度)
    v_spd = status.vSpd
    performance = status.performance

    spd_cmd = control.spdCmd
    if spd_cmd is not None:
        if 120 > now - spd_cmd.assignTime >= 0:
            guidance.targetHSpd = performance.min_max_spd(spd_cmd.targetSpd, v_spd=v_spd)
        elif now - spd_cmd.assignTime >= 120:
            control.transition(mode='Spd')
    else:
        if v_spd == 0.0:
            guidance.targetHSpd = performance.normCruiseTAS
        elif v_spd > 0.0:
            guidance.targetHSpd = performance.normClimbTAS
        else:
            guidance.targetHSpd = performance.normDescentTAS

    # 高度引导
    alt_cmd = control.altCmd
    if alt_cmd is not None and now - alt_cmd.assignTime == 0:
        guidance.targetAlt = alt_cmd.targetAlt
        control.transition(mode='Alt')

    # 航向引导（Dogleg机动）
    hdg_cmd = control.hdgCmd
    if hdg_cmd is None:
        guidance.targetCourse = profile.courseToTarget
        return

    delta, assign_time = hdg_cmd.delta, hdg_cmd.assignTime
    if delta == 0:
        control.transition(mode='Hdg')
        return

    elif now - assign_time == 0:    # 以delta角度出航
        guidance.targetCourse = (delta+status.heading) % 360
    elif now - assign_time == 60:   # 转向后持续60秒飞行，之后以30°角切回航路
        prefix = abs(delta) / delta
        guidance.targetCourse = (-prefix*(abs(delta)+30)+status.heading) % 360
    elif now - assign_time == 120:  # 结束偏置（dogleg机动）
        control.transition(mode='Hdg')
