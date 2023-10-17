import time


class Timer:
    """秒数为单位"""
    def __init__(self):
        self.start_time = None
        self.middle_points = []

    def check_is_start(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not started.")

    def start(self):
        """全局开始"""
        self.start_time = time.time()
        return self

    def middle_point(self):
        """中间点"""
        self.check_is_start()
        self.middle_points.append(time.time())
        return self

    def end(self):
        """全局结束"""
        self.check_is_start()

        time_diff = time.time() - self.start_time

        self.start_time = None
        self.middle_points = []
        return time_diff

    def last_timestamp_diff(self):
        """上一点时间"""
        self.check_is_start()

        if len(self.middle_points) == 0:
            return time.time() - self.start_time

        time_diff = time.time() - self.middle_points[-1]
        return time_diff

    def clear(self):
        """重置Timer"""
        self.check_is_start()

        self.start_time = None
        self.middle_points = []
        return self
