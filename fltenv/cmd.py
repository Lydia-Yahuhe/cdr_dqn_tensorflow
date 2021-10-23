from fltsim.aircraft import atccmd
from fltsim.utils import convert_with_align

CmdCount = 81
KT2MPS = 0.514444444444444
NM2M = 1852
flight_level = [i*300.0 for i in range(29)]
flight_level += [i*300.0 + 200.0 for i in range(29, 50)]


def calc_level(alt, v_spd, delta):
    delta = int(delta / 300.0)
    lvl = int(alt / 300.0) * 300.0

    if alt < 8700.0:
        idx = flight_level.index(lvl)
        if (v_spd > 0 and alt - lvl != 0) or (v_spd == 0 and alt - lvl > 150):
            idx += 1

        return flight_level[idx+delta]

    lvl += 200.0
    idx = flight_level.index(lvl)
    if v_spd > 0 and alt - lvl > 0:
        idx += 1
    elif v_spd < 0 and alt - lvl < 0:
        idx -= 1

    return flight_level[idx+delta]


def check_cmd(cmd, a, alt_check):
    if not a.is_enroute() or not a.next_leg:
        return False, '1'

    if cmd.cmdType == atccmd.ATCCmdType.Heading:
        return True, 'Hdg'

    if cmd.cmdType == atccmd.ATCCmdType.Speed:
        return True, 'Spd'

    if cmd.cmdType == atccmd.ATCCmdType.Altitude:
        # 最高12000m，最低6000m
        target_alt = cmd.targetAlt
        if target_alt > 12000 or target_alt < 6000:
            return False, '2'

        # 下降的航空器不能上升，或上升的航空器不能下降
        v_spd, delta = a.status.vSpd, cmd.delta
        if v_spd * delta < 0:
            return False, '3'

        # 调过上升，又调下降，或调过下降，又调上升
        if delta == 0.0:
            prefix = int(abs(v_spd) / v_spd) if v_spd != 0.0 else 0
        else:
            prefix = int(abs(delta) / delta)

        if prefix == 0:
            return True, '0'

        if len(alt_check) > 0 and prefix not in alt_check:
            return False, '4'
        alt_check.append(prefix)
        return True, '0'

    raise NotImplementedError


def int_2_atc_cmd(time: int, idx: int, target):
    # 将idx转化成三进制数
    [alt_idx, spd_idx, hdg_idx, time_idx] = convert_with_align(idx, x=3, align=4)
    # print(idx, alt_idx, spd_idx, hdg_idx, time_idx)

    # time cmd
    time_cmd = int(time_idx) * 15

    # alt cmd
    delta = (int(alt_idx) - 1) * 600.0
    targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
    alt_cmd = atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time+time_cmd)

    # spd cmd
    delta = (int(spd_idx) - 1) * 10.0 * KT2MPS
    targetSpd = target.status.hSpd + delta
    spd_cmd = atccmd.SpdCmd(delta=delta, targetSpd=targetSpd, assignTime=time+time_cmd)

    # hdg cmd
    delta = (int(hdg_idx) - 1) * 45
    hdg_cmd = atccmd.HdgCmd(delta=delta, assignTime=time+time_cmd)

    return [time_cmd, alt_cmd, spd_cmd, hdg_cmd]

    # 高度调整
    # if idx < 3:  # [0, 2]  ALT: [-300:300:300]
    #     delta = (idx - 1) * 300.0
    #     targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
    #     return atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time), 0
    # if idx < 6:  # [3, 5]  ALT: [-300:300:300]
    #     delta = (idx - 4) * 300.0
    #     targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
    #     return atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time), 15
    # if idx < 9:  # [6, 8]  ALT: [-300:300:300]
    #     delta = (idx - 7) * 300.0
    #     targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
    #     return atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time), 30

    # 航向调整
    # if idx < 12:  # [9, 11]  HDG: [30:15:60]
    #     delta = (idx - 7) * 15
    #     return atccmd.HdgCmd(delta=delta, assignTime=time), 0
    # if idx < 15:  # [12, 14]  HDG: [30:15:60]
    #     delta = (idx - 10) * 15
    #     return atccmd.HdgCmd(delta=delta, assignTime=time), 15
    # if idx < 18:  # [15, 17]  HDG: [30:15:60]
    #     delta = (idx - 13) * 15
    #     return atccmd.HdgCmd(delta=delta, assignTime=time), 30
    # if idx < 21:  # [18, 20]  HDG: [-60:15:-30]
    #     delta = (idx - 22) * 15
    #     return atccmd.HdgCmd(delta=delta, assignTime=time), 0
    # if idx < 24:  # [21, 23]  HDG: [-60:15:-30]
    #     delta = (idx - 25) * 15
    #     return atccmd.HdgCmd(delta=delta, assignTime=time), 15
    # if idx < 27:  # [24, 26]  HDG: [-60:15:-30]
    #     delta = (idx - 28) * 15
    #     return atccmd.HdgCmd(delta=delta, assignTime=time), 30


def reward_for_cmd(cmd_info):
    cmd_list = cmd_info['cmd']
    ok_list = cmd_info['ok']

    if False in ok_list:
        reward = sum([-1.0*int(not ok) for ok in ok_list])
    else:
        # reward = sum([int(cmd.delta != 0.0) * (-0.1) for cmd in cmd_list])
        reward = -0.0

    return reward
