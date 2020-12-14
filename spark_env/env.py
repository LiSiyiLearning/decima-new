import numpy as np
import copy
from collections import OrderedDict
from param import *
from utils import *
from spark_env.action_map import compute_act_map, get_frontier_acts
from spark_env.reward_calculator import RewardCalculator
# from spark_env.moving_executors import MovingExecutors
from spark_env.executor_commit import ExecutorCommit
from spark_env.free_executors import FreeExecutors
from spark_env.job_generator import generate_jobs
from spark_env.wall_time import WallTime
from spark_env.timeline import Timeline
from spark_env.executor import Executor
from spark_env.job_dag import JobDAG
from spark_env.task import Task
from spark_env.wave import Wave
from random import randint



class Environment(object):
    def __init__(self):

        # isolated random number generator
        self.np_random = np.random.RandomState()

        # global timer
        self.wall_time = WallTime()

        # uses priority queue 相当于是实现了一个优先级的map？存储了k-v对应关系 但是感觉这个地方是用来算reward用的？存task的完成时间？
        self.timeline = Timeline()

        # executors 有序集合保证executor不会被重复加入也不会失序 但是为什么要保证不失序 这样就可以用迭代器了？
        # 字典 0-97下标标识每个时刻的executor数
        # cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.curve = np.load("totle_exec.npy", allow_pickle=True)
        self.curve = self.curve.transpose()
        # 这里要添加一个参数，level对应的executor数量
        self.level_range = args.exec_level_range
        self.executors = [OrderedSet() for _ in range(self.level_range)]
        self.base = 0
        for level in range(self.level_range):
            t = self.curve[0][level]
            for exec_id in range(self.base, t + self.base):
                self.executors[level].add(Executor(exec_id, level))
            self.base += t
        # free executors 记载了在哪个job上有空余未运行的executor，包括一个key为none表示全局
        # 修改后是哪个level有空闲
        self.free_executors = FreeExecutors(self.executors, self.level_range)
        self.usingExecutors = [OrderedSet() for _ in range(self.level_range)]
        self.nextTime = 0
        # 添加一个参数 两次变动中间的时间间隔
        self.timeInterval = args.time_interval

        # # moving executors 记录每个executor正在执行的node和每个node的executor轨迹
        # self.moving_executors = MovingExecutors()
        #
        # executor commit  ？？
        # self.exec_commit = ExecutorCommit()

        # prevent agent keeps selecting the same node 防止选中同一个node？
        self.node_selected = set()

        # for computing reward at each step
        self.reward_calculator = RewardCalculator()

    # def add_job(self, job_dag):
    #     # 为了添加一个可能出现的job
    #     self.moving_executors.add_job(job_dag)
    #     # self.free_executors.add_job(job_dag)
    #     self.exec_commit.add_job(job_dag)

    def assign_executor(self, executor):
        if executor.node is not None and not executor.node.no_more_tasks:
            # executor有正在执行的node而且node依然有剩余的task 与每次执行完并行度就回去重新调度是否冲突？是否就代表了是否local
            # 每个task的执行完成时间不同，仍有剩余task未完成就继续工作
            # keep working on the previous node
            task = executor.node.schedule(executor)
            # 记录完成时间与对应的task
            self.timeline.push(task.finish_time, task)
        else:
            # 放去执行新的node
            executor.detach_node()
            self.usingExecutors[executor.level].remove(executor)
            # scheduled = False
            # # 遍历commit检查是否有可以分配的node
            # for job in self.exec_commit:
            #     if not scheduled:
            #         for node in self.exec_commit[job]:
            #             if not scheduled and node.num_tasks - node.next_task_idx > len(node.executors):
            #                 for level in self.exec_commit[job][node]:
            #                     if not scheduled and level == executor.level:
            #                         self.exec_commit.pop(job, node. executor.level)
            #                         task = executor.node.schedule(executor)
            #                         self.timeline.push(task.finish_time, task)
            #                         scheduled = True
            # 将executor分配到node上之后while中可能会调动这个，就将executor放回去
            # if self.next_node is None:
            self.free_executors[executor.level].add(executor)
            # else:
            #     task = self.next_node.schedule(executor)
            #     self.timeline.push(task.finish_time, task)
            # need to move on to other nodes
            # if frontier_changed:
            #     # frontier changed, need to consult all free executors
            #     # note: executor.job_dag might change after self.schedule()
            #     # 上次executor执行的job
            #     # source_job = executor.job_dag
            #     if len(self.exec_commit[executor.node]) > 0:
            #         # directly fulfill the commitment
            #         # 如果当前executor上的node有剩余需要执行的task
            #         self.exec_to_schedule = {executor}
            #         self.schedule()
            #     else:
            #         # free up the executor 上次执行了这个
            #         self.free_executors.add(executor.level, executor)
            #     # then consult all free executors
            #     # # 可以调度的所有executor：上次执行了他的所有空闲executor
            #     # self.exec_to_schedule = OrderedSet(self.free_executors[source_job])
            #     # # 记录刚释放了executor的job
            #     # self.source_job = source_job
            #     # # 刚被释放依然在这个job上的executor数量
            #     # self.num_source_exec = len(self.free_executors[source_job])
            # else:
            #     # just need to schedule one current executor 前置节点未完成，只更新信息
            #     self.exec_to_schedule = {executor}
            #     # only care about executors on the node
            #     if len(self.exec_commit[executor.node]) > 0:
            #         # directly fulfill the commitment
            #         self.schedule()
            #     # else:
            #     #     # need to consult for ALL executors on the node
            #     #     # Note: self.exec_to_schedule is immediate
            #     #     #       self.num_source_exec is for commit
            #     #     #       so len(self.exec_to_schedule) !=
            #     #     #       self.num_source_exec can happen    ?
            #     #     self.source_job = executor.job_dag
            #     #     # 所有正在这个节点上运行的executor数量？
            #     #     self.num_source_exec = len(executor.node.executors)

    # def backup_schedule(self, executor):
    #     # This function is triggered very rarely. A random policy
    #     # or the learned polici in early iterations might decide
    #     # to schedule no executors to any job. This function makes
    #     # sure the cluster is work conservative. Since the backup
    #     # policy is not strong, the learning agent should learn to
    #     # not rely on it.
    #     backup_scheduled = False
    #     # if executor.node is not None:
    #     #     # first try to schedule on current job
    #     #     for node in executor.job_dag.frontier_nodes:
    #     #         if not self.saturated(node):
    #     #             # greedily schedule a frontier node
    #     #             task = node.schedule(executor)
    #     #             self.timeline.push(task.finish_time, task)
    #     #             backup_scheduled = True
    #     #             break
    #     for job_dag in self.job_dags:
    #         for node in job_dag.frontier_nodes:
    #             if not self.saturated(node):
    #                 # greedily schedule a frontier node
    #                 task = node.schedule(executor)
    #                 self.timeline.push(task.finish_time, task)
    #                 backup_scheduled = True
    #                 break
    #     # # then try to schedule on any available node
    #     # if not backup_scheduled:
    #     #     schedulable_nodes = self.get_frontier_nodes()
    #     #     if len(schedulable_nodes) > 0:
    #     #         node = next(iter(schedulable_nodes))
    #     #         # 移动executor
    #     #         self.timeline.push(
    #     #             self.wall_time.curr_time + args.moving_delay, executor)
    #     #         # keep track of moving executors
    #     #         self.moving_executors.add(executor, node)
    #     #         backup_scheduled = True
    #     # at this point if nothing available, leave executor idle
    #     if not backup_scheduled:
    #         self.free_executors.add(executor.level, executor)

    def get_frontier_nodes(self):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        frontier_nodes = OrderedSet()
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                # 未执行过，未调度过
                if not node in self.node_selected and not self.saturated(node):
                    parents_saturated = True
                    for parent_node in node.parent_nodes:
                        if not self.saturated(parent_node):
                            parents_saturated = False
                            break
                    if parents_saturated:
                        frontier_nodes.add(node)

        return frontier_nodes

    def get_executor_limits(self):
        # "minimum executor limit" for each job
        # executor limit := {job_dag -> int}
        # 这个job上正在使用的executor数量 job_dag.executors是所有的job上executor的集合 curr_exec是已经空闲的executor
        # 并行度下限
        executor_limit = {}

        for job_dag in self.job_dags:

            # if self.source_job == job_dag:
            #     curr_exec = self.num_source_exec
            # else:
            #     curr_exec = 0
            #
            # # note: this does not count in the commit and moving executors  ?
            executor_limit[job_dag] = [len(level) for level in job_dag.executors]

        return executor_limit

    def get_different_levels_executor_remain(self):
        return [len(s) for s in self.free_executors]

    def get_executor_num(self):
        return [len(s) for s in self.executors]

    def observe(self):
        return self.job_dags, self.get_frontier_nodes(), self.get_executor_limits(), \
               self.action_map, self.get_different_levels_executor_remain(), self.get_executor_num()#, \
               # self.get_different_levels_executor_remain()# self.source_job, self.num_source_exec, \

    def saturated(self, node):
        # frontier nodes := unsaturated nodes with all parent nodes saturated所有可执行调度的节点
        # ?
        # anticipated_task_idx = node.next_task_idx # + \
        #    self.exec_commit.node_commit[node] + \
        #    self.moving_executors.count(node)
        # note: anticipated_task_idx can be larger than node.num_tasks
        # when the tasks finish very fast before commitments are fulfilled
        return len(node.remain_tasks) == 0

    # def schedule(self):
    #     # 下一个可被调度的executor
    #     # 这一批executor绑定新的node
    #     executor = next(iter(self.exec_to_schedule))
    #     # 如果当前executor有正在执行的节点就schedule这个节点否则schedule这个job
    #     source = executor.job_dag if executor.node is None else executor.node
    #
    #     # schedule executors from the source until the commitment is fulfilled
    #     # 没有等待执行的下一步动作
    #     # 没有可以调度的executor就结束调度
    #     while len(self.exec_commit[source]) > 0 and \
    #           len(self.exec_to_schedule) > 0:
    #
    #         # keep fulfilling the commitment using free executors
    #         # 等待被调度的task
    #         node, level = self.exec_commit.pop(source)
    #         # 选一个executor
    #         executor = self.exec_to_schedule.pop()
    #
    #         # mark executor as in use if it was free executor previously 在此之前没有正在执行的node
    #         if self.free_executors.contain_executor(executor.job_dag, executor):
    #             self.free_executors.remove(executor)
    #
    #         # 下一个执行的node
    #         if node is None:
    #             # the next node is explicitly silent, make executor ilde
    #             # 这个job依然未完成，否则设为空闲
    #             if executor.job_dag is not None and \
    #                any([not n.no_more_tasks for n in \
    #                     executor.job_dag.nodes]):
    #                 # mark executor as idle in its original job
    #                 self.free_executors.add(executor.job_dag, executor)
    #             else:
    #                 # no where to assign, put executor in null pool
    #                 self.free_executors.add(None, executor)
    #
    #
    #         elif not node.no_more_tasks:
    #             # node is not currently saturated
    #             if executor.job_dag == node.job_dag:
    #                 # executor local to the job
    #                 if node in node.job_dag.frontier_nodes:
    #                     # node is immediately runnable
    #                     task = node.schedule(executor)
    #                     self.timeline.push(task.finish_time, task)
    #                 else:
    #                     # put executor back in the free pool 还不能执行
    #                     self.free_executors.add(executor.job_dag, executor)
    #
    #             else:
    #                 # need to move executor 该node已执行完
    #                 self.timeline.push(
    #                     self.wall_time.curr_time + args.moving_delay, executor)
    #                 # keep track of moving executors 已移除
    #                 self.moving_executors.add(executor, node)
    #
    #         else:
    #             # node is already saturated, use backup logic
    #             self.backup_schedule(executor)

    def schedule(self, node, executor):
        if len(node.remain_tasks) == 0:
            return
        task = node.schedule(executor)
        self.usingExecutors[executor.level].add(executor)
        self.timeline.push(task.finish_time, task)
    
    def step(self, next_node, limit, level):

        # print("limit:{}".format(limit))
        #if next_node is not None:
            #print(next_node.next_task_idx, next_node.num_tasks, limit)
        # mark the node as selected
        assert next_node not in self.node_selected
        self.node_selected.add(next_node)
        # self.next_node = next_node
        # # commit the source executor
        # executor = next(iter(self.exec_to_schedule))
        # source = executor.job_dag if executor.node is None else executor.node

        # compute number of valid executors to assign 剩余需要调度的task和limit的最小值
        # if next_node is not None:
        #
        # use_exec = min(next_node.num_tasks - next_node.next_task_idx, limit)
        # # else:
        # #     use_exec = limit
        # assert use_exec > 0
        # 把executor分配给对应的node
        # self.exec_commit.add(next_node.job_dag, next_node, level, limit)
        for i in range(limit):
            executor = self.free_executors.pop(level)
            self.schedule(next_node, executor)
        # 检查actor_agent是否可以继续调度
        if next_node is not None:
            reward = self.reward_calculator.get_reward(
                self.job_dags, self.wall_time.curr_time)
            return self.observe(), reward, (len(self.job_dags) == 0) or \
               (self.wall_time.curr_time >= self.max_time)

        self.node_selected.clear()
        # self.next_node = None
        # 开始清空timeline，进行时间更改
        push = True
        change = False
        while not change and (push or sum([len(level) for level in self.free_executors]) == 0 or len(self.get_frontier_nodes()) == 0 and len(self.timeline) > 0 and len(self.job_dags) > 0):
            #print("step")
            new_time, obj = self.timeline.pop()
            self.wall_time.update_time(new_time)
            # print(new_time)
            while self.wall_time.curr_time == new_time:
                if isinstance(obj, Wave):
                    change = True
                    for level in range(self.level_range):
                        if obj.before[level] > obj.after[level]:
                            # print("decrease")
                            # 减少
                            remove = obj.before[level] - obj.after[level]
                            while remove > 0 and len(self.free_executors[level]) > 0:
                                executor = self.free_executors[level].pop()
                                self.executors[level].remove(executor)
                                remove -= 1
                            if remove > 0:
                                # 仍需减少executor
                                while remove > 0:
                                    executor = self.usingExecutors[level].poplast()
                                    self.timeline.remove(executor.task)
                                    executor.task.rollback()
                                    executor.detach_node()
                                    executor.reset()
                                    self.executors[executor.level].remove(executor)
                                    remove -= 1
                        elif obj.before[level] < obj.after[level]:
                            # 增加
                            add = obj.after[level] - obj.before[level]
                            for idx in range(self.base, self.base + add):
                                executor = Executor(idx, level)
                                self.executors[level].add(executor)
                                self.free_executors.add(level, executor)
                            self.base += add
                    if self.nextTime > 97:
                        after = randint(0, 97)
                    else:
                        after = self.nextTime
                    self.nextTime += 1
                    self.timeline.push(self.wall_time.curr_time + self.timeInterval, Wave(obj.after, self.curve[after]))
                    break

                #print("while")
                elif isinstance(obj, Task):
                    node = obj.node
                    node.num_finished_tasks += 1
                    frontier_changed = False
                    if node.num_finished_tasks >= node.num_tasks:
                        # assert not node.tasks_all_done  # only complete once 防止重复判断
                        node.tasks_all_done = True
                        node.job_dag.num_nodes_done += 1
                        node.node_finish_time = self.wall_time.curr_time
                        frontier_changed = node.job_dag.update_frontier_nodes(node)

                    # assign new destination for the job
                    self.assign_executor(obj.executor)

                    # bookkeepings for job completion 全部完成
                    if node.job_dag.num_nodes_done >= node.job_dag.num_nodes:
                        # assert not node.job_dag.completed  # only complete once
                        node.job_dag.completed = True
                        node.job_dag.completion_time = self.wall_time.curr_time
                        self.remove_job(node.job_dag)

                    if len(self.timeline) > 0:
                        new_time, obj = self.timeline.pop()
                    else:
                        break

            #print("out")
            if not self.wall_time.curr_time == new_time:
                self.timeline.push(new_time, obj)
            push = False


        # 从source移到node上
        # self.exec_commit.add(source, next_node, use_exec)
        # deduct the executors that know the destination  可用的executor数？
        # self.num_source_exec -= use_exec

        # assert self.num_source_exec >= 0

        # source上没有空闲的executor了
        # if self.num_source_exec == 0:
        #     # now a new scheduling round, clean up node selection
        #     self.node_selected.clear()
        #     # all commitments are made, now schedule free executors
        #     self.schedule()

        # Now run to the next event in the virtual timeline
        # while len(self.timeline) > 0 and self.num_source_exec == 0:
        #     # consult agent by putting executors in source_exec
        #
        #     new_time, obj = self.timeline.pop()
        #     self.wall_time.update_time(new_time)
        #
        #     # case task: a task completion event, and frees up an executor.
        #     # case query: a new job arrives
        #     # case executor: an executor arrives at certain job
        #
        #     if isinstance(obj, Task):  # task completion event
        #         finished_task = obj
        #         node = finished_task.node
        #         node.num_finished_tasks += 1
        #
        #         # bookkeepings for node completion
        #         frontier_changed = False
        #         if node.num_finished_tasks == node.num_tasks:
        #             assert not node.tasks_all_done  # only complete once 防止重复判断
        #             node.tasks_all_done = True
        #             node.job_dag.num_nodes_done += 1
        #             node.node_finish_time = self.wall_time.curr_time
        #
        #             frontier_changed = node.job_dag.update_frontier_nodes(node)
        #
        #         # assign new destination for the job
        #         self.assign_executor(finished_task.executor, frontier_changed)
        #
        #         # bookkeepings for job completion 全部完成
        #         if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
        #             assert not node.job_dag.completed  # only complete once
        #             node.job_dag.completed = True
        #             node.job_dag.completion_time = self.wall_time.curr_time
        #             self.remove_job(node.job_dag)
        #
        #     elif isinstance(obj, JobDAG):  # new job arrival event
        #         job_dag = obj
        #         # job should be arrived at the first time job只到达一次
        #         assert not job_dag.arrived
        #         job_dag.arrived = True
        #         # inform agent about job arrival when stream is enabled
        #         self.job_dags.add(job_dag)
        #         self.add_job(job_dag)
        #         # 把调度动作映射成每个节点
        #         self.action_map = compute_act_map(self.job_dags)
        #         # assign free executors (if any) to the new job 有可用executor
        #         if len(self.free_executors[None]) > 0:
        #             self.exec_to_schedule = \
        #                 OrderedSet(self.free_executors[None])
        #             self.source_job = None
        #             self.num_source_exec = \
        #                 len(self.free_executors[None])
        #
        #     elif isinstance(obj, Executor):  # executor arrival event executor执行完了上面的task
        #         executor = obj
        #         # pop destination from the tracking record  ？
        #         node = self.moving_executors.pop(executor)
        #
        #         if node is not None:
        #             # the job is not yet done when executor arrives
        #             executor.job_dag = node.job_dag
        #             node.job_dag.executors.add(executor)
        #
        #         if node is not None and not node.no_more_tasks:
        #             # the node is still schedulable
        #             if node in node.job_dag.frontier_nodes:
        #                 # node is immediately runnable
        #                 task = node.schedule(executor)
        #                 self.timeline.push(task.finish_time, task)
        #             else:
        #                 # free up the executor in this job
        #                 self.free_executors.add(executor.job_dag, executor)
        #         else:
        #             # the node is saturated or the job is done
        #             # by the time the executor arrives, use
        #             # backup logic
        #             self.backup_schedule(executor)
        #
        #     else:
        #         print("illegal event type")
        #         exit(1)
        # print("reward")
        # compute reward
        reward = self.reward_calculator.get_reward(
            self.job_dags, self.wall_time.curr_time)

        # no more decision to make, jobs all done or time is up
        done = (len(self.job_dags) == 0) or \
               (self.wall_time.curr_time >= self.max_time)

        # if done:
        #     #print("done")
        #     assert self.wall_time.curr_time >= self.max_time or \
        #            len(self.job_dags) == 0

        return self.observe(), reward, done

    def remove_job(self, job_dag):
        for level in range(self.level_range):
            for executor in job_dag.executors[level]:
                executor.detach_node()
        # self.exec_commit.remove_job(job_dag)
        # self.free_executors.remove_job(job_dag)
        # self.moving_executors.remove_job(job_dag)
        self.job_dags.remove(job_dag)
        self.finished_job_dags.add(job_dag)
        self.action_map = compute_act_map(self.job_dags)

    def reset(self, max_time=np.inf):
        self.max_time = max_time
        self.wall_time.reset()
        self.timeline.reset()
        # self.exec_commit.reset()
        # self.moving_executors.reset()
        self.reward_calculator.reset()
        self.finished_job_dags = OrderedSet()
        self.node_selected.clear()
        self.base = 0
        for level in range(self.level_range):
            self.executors[level].clear()
            self.usingExecutors[level].clear()
            t = self.curve[0][level]
            for exec_id in range(self.base, t + self.base):
                self.executors[level].add(Executor(exec_id, level))
            self.base += t

        self.free_executors.reset(self.executors)
        # generate a set of new jobs
        
        self.job_dags = generate_jobs(self.np_random,
            self.timeline, self.wall_time)

        # map action to dag_idx and node_idx
        self.action_map = compute_act_map(self.job_dags)
        # add initial set of jobs in the system
        # for job_dag in self.job_dags:
        #     self.add_job(job_dag)
        # # put all executors as source executors initially
        # self.exec_to_schedule = OrderedSet()
        # for executor in self.executors:
        #     self.exec_to_schedule.add(executor)
        self.timeline.push(self.timeInterval, Wave(self.curve[0], self.curve[1]))
        self.nextTime = 2

    def seed(self, seed):
        self.np_random.seed(seed)


