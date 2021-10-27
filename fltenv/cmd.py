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


def check_cmd(cmd, a, check_dict):
    if not a.is_enroute() or not a.next_leg:
        return False, '1'

    if cmd.cmdType == atccmd.ATCCmdType.Heading:
        return True, 'Hdg'

    if cmd.cmdType == atccmd.ATCCmdType.Speed:
        return True, 'Spd'

    if cmd.cmdType == atccmd.ATCCmdType.Altitude:
        check = check_dict['ALT']

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

        if len(check) > 0 and prefix not in check:
            return False, '4'
        check.append(prefix)
        return True, '0'

    raise NotImplementedError


def int_2_atc_cmd(time: int, idx: int, target):
    # # 将idx转化成三进制数
    # [alt_idx, spd_idx, hdg_idx, time_idx] = convert_with_align(idx, x=3, align=4)
    # # print(idx, alt_idx, spd_idx, hdg_idx, time_idx)
    #
    # # time cmd
    # time_cmd = int(time_idx) * 15
    #
    # # alt cmd
    # delta = (int(alt_idx) - 1) * 600.0
    # targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
    # alt_cmd = atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time+time_cmd)
    #
    # # spd cmd
    # delta = (int(spd_idx) - 1) * 10.0 * KT2MPS
    # targetSpd = target.status.hSpd + delta
    # spd_cmd = atccmd.SpdCmd(delta=delta, targetSpd=targetSpd, assignTime=time+time_cmd)
    #
    # # hdg cmd
    # delta = (int(hdg_idx) - 1) * 45
    # hdg_cmd = atccmd.HdgCmd(delta=delta, assignTime=time+time_cmd)
    #
    # return [time_cmd, alt_cmd, spd_cmd, hdg_cmd]

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


def reward_for_cmd(cmd_info):
    reward = 0.0
    for cmd in cmd_info['cmd']:
        if not cmd.ok:
            reward += -0.5
        else:
            reward += int(cmd.delta != 0.0) * (-0.2)
    return reward
