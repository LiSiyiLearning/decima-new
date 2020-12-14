import numpy as np
from param import *
from utils import OrderedSet


class Node(object):
    def __init__(self, idx, tasks, task_duration, wall_time, np_random, cpu, mem):
        self.idx = idx
        self.tasks = tasks
        self.wall_time = wall_time
        self.np_random = np_random

        self.task_duration = task_duration
        self.cpu = cpu
        self.mem = mem
        self.num_tasks = len(tasks)
        self.num_finished_tasks = 0
        # self.next_task_idx = 0
        self.remain_tasks = set(list(range(len(tasks))))

        self.no_more_tasks = False
        self.tasks_all_done = False
        self.node_finish_time = np.inf

        self.executors = OrderedSet()

        # uninitialized
        self.parent_nodes = []
        self.child_nodes = []
        self.descendant_nodes = []
        self.job_dag = None

        self.assign_node_to_tasks()

    def assign_node_to_tasks(self):
        for task in self.tasks:
            task.node = self
            task.cpu = self.cpu
            task.mem = self.mem

    def get_node_duration(self):
        # Warning: this is slow O(num_tasks)
        # get the total duration over all tasks
        duration = 0
        for task in self.tasks:
            duration += task.get_duration()
        return duration

    def is_schedulable(self):
        if self.no_more_tasks:  # no more tasks
            return False
        if self.tasks_all_done:  # node done
            return False
        for node in self.parent_nodes:
            if not node.tasks_all_done:  # a parent node not done
                return False
        return True

    def reset(self):
        for task in self.tasks:
            task.reset()
        self.executors.clear()
        self.num_finished_tasks = 0
        # self.next_task_idx = 0
        self.no_more_tasks = False
        self.tasks_all_done = False
        self.node_finish_time = np.inf
        self.remain_tasks = set(list(range(len(tasks))))


    # def sample_executor_key(self, num_executors):
    #     (left_exec, right_exec) = \
    #         self.job_dag.executor_interval_map[num_executors]
    #
    #     executor_key = None
    #
    #     if left_exec == right_exec:
    #         executor_key = left_exec
    #
    #     else:
    #         rand_pt = self.np_random.randint(1, right_exec - left_exec + 1)
    #         if rand_pt <= num_executors - left_exec:
    #             executor_key = left_exec
    #         else:
    #             executor_key = right_exec
    #
    #     if executor_key not in self.task_duration['first_wave']:
    #         # more executors than number of tasks in the job
    #         largest_key = 0
    #         for e in self.task_duration['first_wave']:
    #             if e > largest_key:
    #                 largest_key = e
    #         executor_key = largest_key
    #
    #     return executor_key

    def schedule(self, executor):
        """执行task，计算duration并将信息记录在task实例中"""
        assert len(self.remain_tasks) <= self.num_tasks
        next_task_idx = self.remain_tasks.pop()

        # task中下一个需要执行的task
        task = self.tasks[next_task_idx]

        # task duration is determined by wave
        # num_executors = len(self.job_dag.executors)
        # assert num_executors > 0

        # sample an executor point in the data 并行度等级
        # executor_key = self.sample_executor_key(num_executors)

        # if executor.task is None or \
        #     executor.task.node.job_dag != task.node.job_dag:
        #     # the executor never runs a task in this job
        #     # fresh executor incurrs a warmup delay 需要迁移
        #     if len(self.task_duration['fresh_durations'][executor_key]) > 0:
        #         # (1) try to directly retrieve the warmup delay from data
        #         fresh_durations = \
        #             self.task_duration['fresh_durations'][executor_key]
        #         i = np.random.randint(len(fresh_durations))
        #         duration = fresh_durations[i]
        #     else:
        #         # (2) use first wave but deliberately add in a warmup delay
        #         first_wave = \
        #             self.task_duration['first_wave'][executor_key]
        #         i = np.random.randint(len(first_wave))
        #         duration = first_wave[i] + args.warmup_delay
        #
        # elif executor.task is not None and \
        #         executor.task.node == task.node and \
        #         len(self.task_duration['rest_wave'][executor_key]) > 0:
        #     # executor was working on this node
        #     # the task duration should be retrieved from rest wave 不需要迁移直接执行
        #     rest_wave = self.task_duration['rest_wave'][executor_key]
        #     i = np.random.randint(len(rest_wave))
        #     duration = rest_wave[i]
        # else:
        #     # executor is fresh to this node, use first wave 是当前节点但是没有rest信息，如果有first就用first替代，否则用fresh替代
        #     if len(self.task_duration['first_wave'][executor_key]) > 0:
        #         # (1) try to retrieve first wave from data
        #         first_wave = \
        #             self.task_duration['first_wave'][executor_key]
        #         i = np.random.randint(len(first_wave))
        #         duration = first_wave[i]
        #     else:
        #         # (2) first wave doesn't exist, use fresh durations instead
        #         # (should happen very rarely)
        #         fresh_durations = \
        #             self.task_duration['fresh_durations'][executor_key]
        #         i = np.random.randint(len(fresh_durations))
        #         duration = fresh_durations[i]

        # detach the executor from old node
        # the executor can run task means it is local
        # to the job at this point 清空上一个执行的node和task，但是记录了job_dag
        executor.detach_node()

        # schedule the task 会更新这个executor记录的相关信息
        task.schedule(self.wall_time.curr_time, self.task_duration, executor)

        self.job_dag.executors[executor.level].add(executor)
        if self.job_dag.start_time == -np.inf:
            self.job_dag.start_time = self.wall_time.curr_time
        # mark executor as running in the node
        self.executors.add(executor)
        executor.node = self

        self.no_more_tasks = len(self.remain_tasks) == 0

        if self.no_more_tasks:
            if self in self.job_dag.frontier_nodes:
                # 类似前置需要完成的节点？
                self.job_dag.frontier_nodes.remove(self)

        # 即本次调度的task
        return task


class NodeDuration(object):
    # A light-weighted extra storage for node duration

    def __init__(self, node):
        self.node = node

        self.task_idx = 0  # next unscheduled task index
        self.duration = self.node.get_node_duration()

        # uninitialized when node is created
        # but can be initialized when job_dag is created
        self.descendant_work = 0  # total work of descedent nodes
        self.descendant_cp = 0    # critical path of descdent nodes


def dfs_nodes_order_by_id(node, nodes_order):
    # Depth first search by node id, use recursive search
    # this is for faithfully reproduce spark scheduling logic
    parent_id = []
    parent_map = {}
    for n in node.parent_nodes:
        parent_id.append(n.idx)
        parent_map[n.idx] = n
    parent_id = sorted(parent_id)
    for i in parent_id:
        dfs_nodes_order_by_id(parent_map[i], nodes_order)
    if node.idx not in nodes_order:
        nodes_order.append(node.idx)
