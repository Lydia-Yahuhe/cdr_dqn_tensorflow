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
    cmdType = ATCCmdType.Altitude

    def __str__(self):
        return 'ALTCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetAlt)

    def __repr__(self):
        return 'ALTCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetAlt)


@dataclass
class SpdCmd(ATCCmd):
    delta: float
    currentSpd: float = 0.0
    targetSpd: float = 0.0
    assignTime: int = 0
    cmdType = ATCCmdType.Speed

    def __str__(self):
        return 'SPDCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetSpd)

    def __repr__(self):
        return 'SPDCMD: <TIME:%d, DELTA:%0.2f, TARGET:%0.2f>' % (self.assignTime, self.delta, self.targetSpd)


@dataclass
class HdgCmd(ATCCmd):
    delta: float
    currentHdg: float = 0.0
    targetHdg: float = 0.0
    assignTime: int = 0
    cmdType = ATCCmdType.Heading

    def __str__(self):
        return 'OFFSET: <TIME: %d, DELTA:%0.2f, TARGET: %d>' % (self.assignTime, self.delta, self.targetHdg)

    def __repr__(self):
        return 'OFFSET: <TIME: %d, DELTA:%0.2f, TARGET: %d>' % (self.assignTime, self.delta, self.targetHdg)
