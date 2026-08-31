"""Wall-clock signal sequencing for the LIVE demo (not a hardware controller)."""
from __future__ import annotations

from network_env import Phase_NS, Phase_EW, MIN_GREEN


class SignalSequencer:
    """A request can change the destination, never bypass an active clearance.

    Deadlines start when the stage is actually displayed. A delayed tick cannot
    skip yellow or all-red, and timing edits cannot shorten an active stage.
    """
    def __init__(self, now, yellow=3, all_red=2):
        self.phase = Phase_NS
        self.stage = "ALL_RED"
        self.started = now
        self.deadline = now + all_red
        self.desired = Phase_NS
        self.yellow = yellow
        self.all_red = all_red

    def advance(self, now, desired):
        self.desired = desired
        if self.stage == "GREEN":
            if desired != self.phase and (desired is None or now - self.started >= MIN_GREEN):
                self.stage, self.started = "YELLOW", now
                self.deadline = now + self.yellow
        elif self.stage == "YELLOW":
            if now >= self.deadline:
                self.stage, self.started = "ALL_RED", now
                self.deadline = now + self.all_red
        elif now >= self.deadline and desired is not None:
            self.phase = desired
            self.stage, self.started = "GREEN", now
            self.deadline = None

    def snapshot(self, now):
        prefix = "NS" if self.phase == Phase_NS else "EW"
        state = "ALL_RED" if self.stage == "ALL_RED" else f"{prefix}_{self.stage}"
        return {
            "phase": state,
            "ns": self.stage if self.phase == Phase_NS and self.stage != "ALL_RED" else "RED",
            "ew": self.stage if self.phase == Phase_EW and self.stage != "ALL_RED" else "RED",
            "time_in_phase": round(max(0, now - self.started), 1),
            "transition_remaining_s": round(max(0, self.deadline - now), 1) if self.deadline is not None else 0,
            "target": "ALL_RED" if self.desired is None else ("NS_GREEN" if self.desired == Phase_NS else "EW_GREEN"),
        }
