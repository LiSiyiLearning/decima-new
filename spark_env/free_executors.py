from utils import OrderedSet


class FreeExecutors(object):
    # 要把这个job为key改成level为key
    def __init__(self, executors, level_range):
        self.level = level_range
        self.free_executors = [OrderedSet() for _ in range(self.level)]
        for level in range(level_range):
            for executor in executors[level]:
                self.free_executors[executor.level].add(executor)

    def __getitem__(self, level):
        return self.free_executors[level]

    def contain_executor(self, level, executor):
        if executor in self.free_executors[level]:
            return True
        else:
            return False

    def pop(self, level):
        executor = next(iter(self.free_executors[level]))
        self.free_executors[level].remove(executor)
        return executor

    def add(self, level, executor):
        # if job is None:
        #     executor.detach_job()
        # else:
        #     executor.detach_node()
        executor.detach_node()
        self.free_executors[level].add(executor)

    def remove(self, executor):
        self.free_executors[executor.level].remove(executor)

    # def add_job(self, job):
    #     self.free_executors[job] = OrderedSet()
    #
    # def remove_job(self, job):
    #     # put all free executors to global free pool
    #     for executor in self.free_executors[job]:
    #         executor.detach_job()
    #         self.free_executors[None].add(executor)
    #     del self.free_executors[job]

    def reset(self, executors):
        # self.free_executors = {}
        for level in range(self.level):
            self.free_executors[level].clear()
            for executor in executors[level]:
                self.free_executors[executor.level].add(executor)
        # self.free_executors[None] = OrderedSet()
        # for executor in executors:
        #     self.free_executors[None].add(executor)
