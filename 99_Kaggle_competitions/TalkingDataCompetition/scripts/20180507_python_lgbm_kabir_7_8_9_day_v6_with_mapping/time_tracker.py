from datetime import datetime

class TimeTracker:
    start_time = None
    last_update_time = None

    def start(self):
        self.start_time = datetime.now()
        self.last_update_time = self.start_time

    def get_time_from_start(self):
        current_time = datetime.now()
        return ((current_time - self.start_time).seconds)

    def update(self):
        current_time = datetime.now()
        temp_time = self.last_update_time
        self.last_update_time = current_time
        return ((current_time - temp_time).seconds)