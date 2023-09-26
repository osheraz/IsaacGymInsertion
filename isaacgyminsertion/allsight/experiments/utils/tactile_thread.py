# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import threading

import pybullet as p

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time


class SoftRealTimeClock:
    """
    Convenience class for sleeping in a loop at a specified rate
    """

    def __init__(self, hz=None, period=None):
        assert (
            hz is not None or period is not None
        ), "Use either SoftRealTimeClock(hz=10) or SoftRealTimeClock(period=0.1)"
        self.last_time = self.gettime()
        self.sleep_dur = 1.0 / hz if hz is not None else period

    def gettime(self):
        return time.clock_gettime(time.CLOCK_REALTIME)

    def _remaining(self, curr_time):
        """
        Calculate the time remaining for clock to sleep.
        """
        elapsed = curr_time - self.last_time
        return self.sleep_dur - elapsed

    def _sleep(self, duration):
        if duration < 0:
            return
        time.sleep(duration)

    def sleep(self):
        """
        Attempt sleep at the specified rate.
        """
        curr_time = self.gettime()
        self._sleep(self._remaining(curr_time))
        self.last_time += self.sleep_dur


def test_soft_real_time_clock():
    clock = SoftRealTimeClock(100)
    for i in range(200):
        print(clock.gettime())
        clock.sleep()

    clock = SoftRealTimeClock(period=0.1)
    for i in range(20):
        print(clock.gettime())
        clock.sleep()



class TactileThread(threading.Thread):
    def __init__(self, real_time_factor, allsight):
        super().__init__()
        self.real_time_factor = real_time_factor
        self.allsight = allsight

    def run(self):
        """
        Use a soft real-time clock (Soft RTC) to step through the simulation.If
        the p.stepSimulation takes too long, slow down the clock by decreasing
        the real time factor.
        """
        time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]
        clock = SoftRealTimeClock(period=time_step / self.real_time_factor)
        while threading.main_thread().is_alive():
            p.stepSimulation()
            color, depth = self.allsight.render()
            self.allsight.updateGUI(color, depth)
            clock.sleep()
