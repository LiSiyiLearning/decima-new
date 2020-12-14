import numpy as np
import traceback

class Task(object):
    def __init__(self, idx, rough_duration, wall_time):
        self.idx = idx
        self.wall_time = wall_time

        self.duration = rough_duration

        # uninitialized
        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None
        self.node = None
        self.cpu = None
        self.mem = None

    def schedule(self, start_time, duration, executor):
        assert np.isnan(self.start_time)
        assert np.isnan(self.finish_time)
        assert self.executor is None

        self.start_time = start_time
        self.duration = duration
        self.finish_time = self.start_time + duration

        # bind the executor to the task and
        # the task with the given executor
        self.executor = executor
        self.executor.task = self
        self.executor.node = self.node

    def get_duration(self):
        # get task duration lazily
        if np.isnan(self.start_time):
            # task not scheduled yet
            return self.duration
        elif self.wall_time.curr_time < self.start_time:
            # task not started yet
            return self.duration
        else:
            # task running or completed
            duration = max(0,
                self.finish_time - self.wall_time.curr_time)
            return duration

    def rollback(self):
        # print(self.node.next_task_idx)
        # print(self.start_time)
        # print(self.idx)
        self.node.remain_tasks.add(self.idx)
        if self.node.no_more_tasks:
            self.node.no_more_tasks = False
            self.node.job_dag.frontier_nodes.add(self.node)
        self.reset()
        # print(self.node.next_task_idx)
        # print(self.start_time)
        # print(self.idx)

    def reset(self):
        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None
