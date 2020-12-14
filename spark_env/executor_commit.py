from collections import OrderedDict


class ExecutorCommit(object):
    def __init__(self):
        # {node/job_dag -> ordered{node -> amount}}
        self.commit = {}
        # # {node -> amount}
        # self.node_commit = {}
        # # {node -> set(nodes/job_dags)}
        # self.backward_map = {}

    # def __getitem__(self, source):
    #     return self.commit[source]
    def __getitem__(self, job):
        return self.commit[job]

    def add(self, job, node, level, limit):
        # source can be node or job
        # node: executors continuously free up
        # job: free executors
        # job指node所属job
        # limit指并行度上限
        # remain指剩余需要分配的executor数
        # add foward connection
        if job not in self.commit:
            self.commit[job] = {"limit": 0}
        assert limit > self.commit[job]["limit"]
        add_exec = limit - self.commit[job]["limit"]
        if node not in self.commit[job]:
            self.commit[job][node] = {}
        if level not in self.commit[job][node]:
            self.commit[job][node][level] = 0

        # add to record of total commit on node
        # 剩余需要的executor增加量为新并行度与原并行度的差
        self.commit[job][node][level] += add_exec
        # add node commit
        # # add backward connection
        # self.backward_map[node].add(source)

    def pop(self, job, node, level):
        # implicitly assert source in self.commit
        # implicitly assert len(self.commit[source]) > 0

        # # find the node in the map
        # node = next(iter(self.commit[source]))
        # level = next(iter(self.commit[source][node]))
        # # deduct one commitment
        # self.commit[source][node][level] -= 1
        # self.node_commit[node][level] -= 1
        # assert self.commit[source][node][level] >= 0
        # assert self.node_commit[node][level] >= 0

        # # remove commitment on job if exhausted
        # if self.commit[source][node][level] == 0:
        #     del self.commit[source][node][level]
        # if len(self.commit[source][node]) == 0:
        #     self.backward_map[node].remove(source)
        assert self.commit[job][node][level] > 0
        self.commit[job][node][level] -= 1
        # 在该level上该node没有需要添加的executor
        if self.commit[job][node][level] == 0:
            del self.commit[job][node][level]
        # 在该node上没有需要添加的executor
        if len(self.commit[job][node]) == 0:
            del self.commit[job][node]
        # return node, level

    # def add_job(self, job_dag):
    #     # add commit entry to the map
    #     self.commit[job_dag] = OrderedDict()
    #     for node in job_dag.nodes:
    #         self.commit[node] = OrderedDict()
    #         self.node_commit[node] = {}
    #         self.backward_map[node] = set()

    def remove_job(self, job_dag):
        # when removing jobs, the commiment should be all satisfied
        assert len(self.commit[job_dag]) == 0
        del self.commit[job_dag]

        # clean up commitment to the job
        # for node in job_dag.nodes:
        #     # the executors should all move out
        #     assert len(self.commit[node]) == 0
        #     del self.commit[node]
        #
        #     for source in self.backward_map[node]:
        #         # remove forward link
        #         del self.commit[source][node]
        #     # remove backward link
        #     del self.backward_map[node]
        #     # remove node commit records
        #     del self.node_commit[node]

    def reset(self):
        self.commit = {}
        # self.node_commit = {}
        # self.backward_map = {}
        # for agent to make void action
        # self.commit[None] = OrderedDict()
        # self.node_commit[None] = 0
        # self.backward_map[None] = set()
