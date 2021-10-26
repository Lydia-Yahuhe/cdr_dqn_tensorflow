from enum import Enum
from dataclasses import dataclass

ATCCmdType = Enum('ATCCmdType', ('Speed', 'Altitude', 'Heading'))


class ATCCmd:
    pass


@dataclass
class AltCmd(ATCCmd):
    delta: float
    currentAlt: float = 0.0
    targetAlt: float = 0.0
    assignTime: int = 0
    ok: bool = True
    cmdType = ATCCmdType.Altitude

    def to_dict(self):
        # return {'Time': self.assignTime, 'Type': self.cmdType,
        #         'Detail': '{}, {}, {}'.format(self.delta, self.targetAlt, self.ok)}
        return {'ALT': '{},{},{}'.format(self.delta, round(self.targetAlt), self.ok)}

    def tostring(self):
        return '{},{},{},{}'.format(self.cmdType, self.delta, round(self.targetAlt), self.ok)

    def __str__(self):
        return 'ALTCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetAlt)

    def __repr__(self):
        return 'ALTCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetAlt)


@dataclass
class SpdCmd(ATCCmd):
    delta: float
    assignTime: int = 0
    ok: bool = True
    currentSpd: float = 0.0
    targetSpd: float = 0.0
    cmdType = ATCCmdType.Speed

    def to_dict(self):
        # return {'Time': self.assignTime, 'Type': self.cmdType,
        #         'Detail': '{}, {}, {}'.format(self.delta, self.targetSpd, self.ok)}
        return {'SPD': '{},{},{}'.format(round(self.delta, 1), round(self.targetSpd), self.ok)}

    def tostring(self):
        return '{},{},{},{}'.format(self.cmdType, self.delta, round(self.targetSpd), self.ok)

    def __str__(self):
        return 'SPDCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetSpd)

    def __repr__(self):
        return 'SPDCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetSpd)


@dataclass
class HdgCmd(ATCCmd):
    delta: float
    assignTime: int = 0
    ok: bool = True
    currentHdg: float = 0.0
    targetHdg: float = 0.0
    cmdType = ATCCmdType.Heading

    def to_dict(self):
        # return {'Time': self.assignTime, 'Type': self.cmdType,
        #         'Detail': '{}, {}, {}'.format(self.delta, round(self.targetHdg, 1), self.ok)}
        return {'HDG': '{},{},{}'.format(self.delta, round(self.targetHdg), self.ok)}

    def tostring(self):
        return '{},{},{},{}'.format(self.cmdType, self.delta, round(self.targetHdg, 1), self.ok)

    def __str__(self):
        return 'OFFSET: <TIME: %d, DELTA:%0.2f, TARGET: %d>' % (self.assignTime, self.delta, self.targetHdg)

    def __repr__(self):
        return 'OFFSET: <TIME: %d, DELTA:%0.2f, TARGET: %d>' % (self.assignTime, self.delta, self.targetHdg)
