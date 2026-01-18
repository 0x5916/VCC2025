import time

class FPSCounter:
    def __init__(self):
        self.fps_counter = 0
        self.last_time = time.time()
        self.current_time = self.last_time + 0.00001
        self.fps_display = 0

    def update_and_get(self):
        self.fps_counter += 1
        self.current_time = time.time()
        if self.current_time - self.last_time >= 1.0:
            self.fps_display = self.fps_counter / (self.current_time - self.last_time)
            self.fps_counter = 0
            self.last_time = self.current_time
        return self.fps_display

class LabelCounter:
    def __init__(self, labels: list[str]):
        self.counter = {}

        for label in labels:
            self.counter[label] = 0

    def update(self, label: str):
        self.counter[label] += 1

    def get_count(self, label: str) -> int:
        return self.counter[label]
