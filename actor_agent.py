import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
import bisect
from param import *
from utils import *
from tf_op import *
from msg_passing_path import *
from gcn import GraphCNN
from gsn import GraphSNN
from agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node


# type_num=[2,3,4,5]


#args传来executor_levels 所有种类中的最大值
# type_num 每个种类的具体数值，如cup,内存等

# 内存cpu当前的策略是直接加起来了，作为网络输入


class ActorAgent(Agent):
    def __init__(self, sess, node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, type_num, exec_mem,exec_num,eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent'): #加了type_num

        Agent.__init__(self)

        self.sess = sess
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.type_num = type_num
        self.exec_mem = exec_mem
        self.exec_num = exec_num
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        # for computing and storing message passing path
        self.postman = Postman()

        # node input dimension: [total_num_nodes, num_features]
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])

        # job input dimension: [total_num_jobs, num_features]
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])

        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)

        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn, self.scope)

        # valid mask for node action ([batch_size, total_num_nodes])
        self.node_valid_mask = tf.placeholder(tf.float32, [None, None])

        # 执行者种类的掩码
        self.type_valid_mask = tf.placeholder(tf.float32, [None, None])

        # valid mask for executor limit on jobs ([batch_size, num_jobs * num_exec_limits])
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])

        # map back the dag summeraization to each node ([total_num_nodes, num_dags])
        self.dag_summ_backward_map = tf.placeholder(tf.float32, [None, None])

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        # 预测结果，加上了执行者选择种类
        self.node_act_probs, self.job_act_probs, self.type_act_probs= self.actor_network(
            self.node_inputs, self.gcn.outputs, self.job_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1],
            self.node_valid_mask, self.job_valid_mask,self.type_valid_mask,
            self.dag_summ_backward_map, self.act_fn)

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        logits = tf.log(self.node_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.node_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # job_acts [batch_size, num_jobs, 1]
        logits = tf.log(self.job_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.job_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)

        # type_acts得出结论 [batch_size, num_jobs, 1]
        logits = tf.log(self.type_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.type_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)




        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])
        self.job_act_vec = tf.placeholder(tf.float32, [None, None, None])
        #  关于种类的，维度暂时未知  train里也要改
        self.type_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # advantage term (from Monte Calro or critic) ([batch_size, 1]) reward-baseline
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ()) #一个参数，类似于未来value的折扣，一般这种都是参数

        # select node action probability
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.node_act_probs, self.node_act_vec),
            reduction_indices=1, keep_dims=True)

        # select job action probability
        self.selected_job_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.job_act_probs, self.job_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        # 选择种类
        self.selected_type_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.type_act_probs, self.type_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        # actor loss due to advantge (negated) tf.multiply（）两个矩阵中对应元素各自相乘 ?
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_node_prob * self.selected_job_prob*self.selected_type_prob + \
            self.eps), -self.adv))

        # node_entropy 信息熵
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.node_act_probs, tf.log(self.node_act_probs + self.eps)))

        # prob on each job
        self.prob_each_job = tf.reshape(
            tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0],
                tf.reshape(self.node_act_probs, [-1, 1])),
                [tf.shape(self.node_act_probs)[0], -1])

        # job entropy
        self.job_entropy = \
            tf.reduce_sum(tf.multiply(self.prob_each_job,
            tf.reduce_sum(tf.multiply(self.job_act_probs,
                tf.log(self.job_act_probs + self.eps)), reduction_indices=2)))

        #type
        self.type_entropy = \
            tf.reduce_sum(tf.multiply(self.prob_each_job,
            tf.reduce_sum(tf.multiply(self.type_act_probs,
                tf.log(self.type_act_probs + self.eps)), reduction_indices=2)))


        # entropy loss
        self.entropy_loss = self.node_entropy + self.job_entropy + self.type_entropy

        # normalize entropy
        self.entropy_loss /= \
            (tf.log(tf.cast(tf.shape(self.node_act_probs)[1], tf.float32)) + \
            tf.log(float(len(self.executor_levels)))+tf.log(float(len(self.type_num)))) #这里加一个种类总数,然后取log
            # normalize over batch size (note: adv_loss is sum)
            # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters 网络的参数
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, node_inputs, gcn_outputs, job_inputs,
                      gsn_dag_summary, gsn_global_summary,
                      node_valid_mask, job_valid_mask, type_valid_mask,
                      gsn_summ_backward_map, act_fn):

        # takes output from graph embedding and raw_input from environment

        batch_size = tf.shape(node_valid_mask)[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = tf.reshape(
            node_inputs, [batch_size, -1, self.node_input_dim])

        # (2) reshape job inputs to batch format
        job_inputs_reshape = tf.reshape(
            job_inputs, [batch_size, -1, self.job_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = tf.reshape(
            gcn_outputs, [batch_size, -1, self.output_dim])

        # (5) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = tf.reshape(
            gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(
            tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_dag_summ_extend = tf.matmul(
            gsn_summ_backward_map_extend, gsn_dag_summ_reshape)

        # (6) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = tf.reshape(
            gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])
        gsn_global_summ_extend_node = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_extend)[1], 1])

        # (4) actor neural network
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            # print(node_inputs_reshape.shape)
            # print(gcn_outputs_reshape.shape)
            # print(gsn_dag_summ_extend.shape)
            # print(gsn_global_summ_extend_node.shape)
            merge_node = tf.concat([
                node_inputs_reshape, gcn_outputs_reshape,
                gsn_dag_summ_extend,
                gsn_global_summ_extend_node], axis=2)

            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)
            

            # reshape the output dimension (batch_size, total_num_nodes)
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])

            # valid mask on node
            node_valid_mask = (node_valid_mask - 1) * 10000.0

            # apply mask
            node_outputs = node_outputs + node_valid_mask

            # do masked softmax over nodes on the graph
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)

            # -- part B, the distribution over executor limits --
            merge_job = tf.concat([
                job_inputs_reshape,
                gsn_dag_summ_reshape,
                gsn_global_summ_extend_job], axis=2)

            expanded_state = expand_act_on_state(  #改成当前的状态
                merge_job, [l / 50.0 for l in self.executor_levels])

            job_hid_0 = tl.fully_connected(expanded_state, 32, activation_fn=act_fn)
            job_hid_1 = tl.fully_connected(job_hid_0, 16, activation_fn=act_fn)
            job_hid_2 = tl.fully_connected(job_hid_1, 8, activation_fn=act_fn)
            job_outputs = tl.fully_connected(job_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])

            # valid mask on job
            job_valid_mask = (job_valid_mask - 1) * 10000.0

            # apply mask
            job_outputs = job_outputs + job_valid_mask

            # reshape output dimension for softmaxing the executor limits
            # (batch_size, num_jobs, num_exec_limits)
            job_outputs = tf.reshape( #改成应该有的并行度
                job_outputs, [batch_size, -1, len(self.executor_levels)])

            # do masked softmax over jobs
            job_outputs = tf.nn.softmax(job_outputs, dim=-1)

            # -- part C, 分配的执行者等级 --
            merge_type = tf.concat([
                job_inputs_reshape,
                gsn_dag_summ_reshape,
                gsn_global_summ_extend_job], axis=2)# 这里要决策需要使用哪些节点来连接网络的输入，应该是要加上node节点的，是node层面的, 和选择节点策略一样，用reward区分

            #该函数用来将mem情况添加到输入里,50是归一化
            expanded_state1 = expand_act_on_state_type( #我们用什么来表示资源的类型 就是l / num_e怎么取
                merge_type, [l / 5 for l in self.type_num], [l / 5 for l in self.exec_mem]) # 假设5是种类数,除数要归一化

            type_hid_0 = tl.fully_connected(expanded_state1, 32, activation_fn=act_fn)
            type_hid_1 = tl.fully_connected(type_hid_0, 16, activation_fn=act_fn)
            type_hid_2 = tl.fully_connected(type_hid_1, 8, activation_fn=act_fn)
            type_outputs = tl.fully_connected(type_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
            type_outputs = tf.reshape(type_outputs, [batch_size, -1])

            # valid mask on node
            type_valid_mask = (type_valid_mask - 1) * 10000.0
            #规格个数个数组/1个数组

            # apply mask
            type_outputs = type_outputs + type_valid_mask
            # reshape output dimension for softmaxing the executor limits
            # (batch_size, num_jobs, num_exec_limits)
            type_outputs = tf.reshape(
                type_outputs, [batch_size, -1, len(self.type_num)]) # 种类数

            # do masked softmax over nodes on the graph
            type_outputs = tf.nn.softmax(type_outputs, dim=-1)

            return node_outputs, job_outputs, type_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run([self.gsn.summaries],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + self.gsn.summ_mats,
                [node_inputs] + summ_mats)
        })

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self, node_inputs, job_inputs, #增加了两个与类别有关的参数
            node_valid_mask, job_valid_mask, type_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map,
            node_act_vec, job_act_vec, type_act_vec,adv, entropy_weight):

        return self.sess.run([self.act_gradients,
            [self.adv_loss, self.entropy_loss]],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.job_inputs] + \
                [self.node_valid_mask] + [self.job_valid_mask] + [self.type_valid_mask]+\
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map] + [self.node_act_vec] + \
                [self.job_act_vec] + [self.type_act_vec] + [self.adv] + [self.entropy_weight], \
                [node_inputs] + [job_inputs] + \
                [node_valid_mask] + [job_valid_mask] + [type_valid_mask] + \
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map] + [node_act_vec] + \
                [job_act_vec] + [type_act_vec] + [adv] + [entropy_weight])
        })

    def predict(self, node_inputs, job_inputs,  # 增加了一些参数
            node_valid_mask, job_valid_mask, type_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map):
        return self.sess.run([self.node_act_probs, self.job_act_probs, self.type_act_probs,
            self.node_acts, self.job_acts, self.type_acts], \
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.job_inputs] + \
                [self.node_valid_mask] + [self.job_valid_mask] + [self.type_valid_mask] +\
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map], \
                [node_inputs] + [job_inputs] + \
                [node_valid_mask] + [job_valid_mask] +  [type_valid_mask] +\
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map])
        })

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def translate_state(self, obs): #等fature改
        """
        Translate the observation to matrix form
        """
        job_dags, \
        frontier_nodes, executor_limits, \
        action_map,get_different_levels_executor_remain,get_executor_num = obs
        #print(get_executor_num)
        # compute total number of nodes
        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags), self.job_input_dim])

        # sort out the exec_map
        exec_map = {}
        
        for job_dag in job_dags:
           
            exec_map[job_dag]={}
            for i in range(len(job_dag.executors)):
                exec_map[job_dag][i]=len(job_dag.executors[i])
        # count in moving executors
        # for node in moving_executors.moving_executors.values():
        #     exec_map[node.job_dag] += 1
        # # count in executor commit
        # for s in exec_commit.commit:
        #     if isinstance(s, JobDAG):
        #         j = s
        #     elif isinstance(s, Node):
        #         j = s.job_dag
        #     elif s is None:
        #         j = None
        #     else:
        #         print('source', s, 'unknown')
        #         exit(1)
        #     for n in exec_commit.commit[s]:
        #         if n is not None and n.job_dag != j:
        #             exec_map[n.job_dag] += exec_commit.commit[s][n]

        # gather job level inputs
        job_idx = 0
        for job_dag in job_dags:
            # number of executors in the job

            #job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
            #job_inputs[job_idx, 0] = 5.0
            # the current executor belongs to this job or not
            # if job_dag is source_job:
            #     job_inputs[job_idx, 1] = 2
            # else:
            #     job_inputs[job_idx, 1] = -2
            # # number of source executors
            # job_inputs[job_idx, 2] = num_source_exec / 20.0

            #有多少个exec剩余，job上的exec都是n个 暂定2 确定怎么归一化

            job_inputs[job_idx, 0] = exec_map[job_dag][0]
            job_inputs[job_idx, 1] = exec_map[job_dag][1]
            job_inputs[job_idx, 2] = exec_map[job_dag][2]
            job_inputs[job_idx, 3] = exec_map[job_dag][3]
            job_inputs[job_idx, 4] = exec_map[job_dag][4]
            job_inputs[job_idx, 5] = exec_map[job_dag][5]
            job_inputs[job_idx, 6] = exec_map[job_dag][6]
            job_inputs[job_idx, 7] = exec_map[job_dag][7]
            job_inputs[job_idx, 8] = exec_map[job_dag][8]
            job_inputs[job_idx, 9] = exec_map[job_dag][9]
            job_inputs[job_idx, 10] = exec_map[job_dag][10]
            job_inputs[job_idx, 11] = exec_map[job_dag][11]
            job_inputs[job_idx, 12] = exec_map[job_dag][12]
            job_inputs[job_idx, 13] = exec_map[job_dag][13]
            job_inputs[job_idx, 14] = exec_map[job_dag][14]
            job_inputs[job_idx, 15] = exec_map[job_dag][15]

            job_inputs[job_idx, 16] = get_different_levels_executor_remain[0]
            job_inputs[job_idx, 17] = get_different_levels_executor_remain[1]
            job_inputs[job_idx, 18] = get_different_levels_executor_remain[2]
            job_inputs[job_idx, 19] = get_different_levels_executor_remain[3]
            job_inputs[job_idx, 20] = get_different_levels_executor_remain[4]
            job_inputs[job_idx, 21] = get_different_levels_executor_remain[5]
            job_inputs[job_idx, 22] = get_different_levels_executor_remain[6]
            job_inputs[job_idx, 23] = get_different_levels_executor_remain[7]
            job_inputs[job_idx, 24] = get_different_levels_executor_remain[8]
            job_inputs[job_idx, 25] = get_different_levels_executor_remain[9]
            job_inputs[job_idx, 26] = get_different_levels_executor_remain[10]
            job_inputs[job_idx, 27] = get_different_levels_executor_remain[11]
            job_inputs[job_idx, 28] = get_different_levels_executor_remain[12]
            job_inputs[job_idx, 29] = get_different_levels_executor_remain[13]
            job_inputs[job_idx, 30] = get_different_levels_executor_remain[14]
            job_inputs[job_idx, 31] = get_different_levels_executor_remain[15]






            job_idx += 1

        # gather node level inputs
        node_idx = 0
        job_idx = 0
        for job_dag in job_dags: # 这里是feature的部分对应论文里5个,我们现在得把fature加上，修改超参数，归一化可以写死
            for node in job_dag.nodes:

                # copy the feature from job_input first
                node_inputs[node_idx, :32] = job_inputs[job_idx, :32]

                # work on the node 这个是什么
                # node_inputs[node_idx, 4] = \
                #     (node.num_tasks - node.next_task_idx) * \
                #     node.tasks[-1].duration / 100000.0
                node_inputs[node_idx, 32] = len(node.executors)

                # number of tasks left
                node_inputs[node_idx, 33] = \
                    len(node.remain_tasks) / 200.0

                #task执行时间
                node_inputs[node_idx, 34] = node.task_duration

                #cpu
                node_inputs[node_idx, 35] = node.cpu
                #mem
                node_inputs[node_idx, 36] = node.mem

                node_idx += 1

            job_idx += 1


        return node_inputs, job_inputs, \
               job_dags, \
               frontier_nodes, executor_limits, \
               exec_map, action_map, get_different_levels_executor_remain,get_executor_num

    def get_valid_masks(self, job_dags, frontier_nodes,
              exec_map, action_map,get_different_levels_executor_remain,get_executor_num): # 加上type_valid_masks，计算掩码

        job_valid_mask = np.zeros([1, len(job_dags) * len(self.executor_levels)])

        type_valid_mask = np.ones([1,len(job_dags) * len(self.type_num)]) #确定type的掩码的维度

        based = 0
        for job_dag in job_dags: #设置type mask
            for l in range(len(self.type_num)):
                if get_different_levels_executor_remain[l]==0:
                    type_valid_mask[0, based + l] = 0
            based += len(self.type_num)
        
        


        job_valid = {}  # if job is saturated, don't assign node

        base = 0
        for job_dag in job_dags:
            least_exec_amount={}
            for i in range(len(exec_map[job_dag])):
                least_exec_amount[i] = exec_map[job_dag][i] + 1
            # # new executor level depends on the source of executor
            # if job_dag is source_job:
            #     least_exec_amount = \
            #         exec_map[job_dag] - num_source_exec + 1
            #         # +1 because we want at least one executor
            #         # for this job
            # else:
            #     least_exec_amount = exec_map[job_dag] + 1
            #     # +1 because of the same reason above

            #assert least_exec_amount > 0
            #assert least_exec_amount <= self.executor_levels[-1] + 1

            # find the index for first valid executor limit
            exec_level_idx={}
            for i in range(len(exec_map[job_dag])):
                # print(self.exec_num[i])
                # print(least_exec_amount[i])
                exec_level_idx[i] = bisect.bisect_left(
                    range(1, get_executor_num[i] + 1), least_exec_amount[i])

                if exec_level_idx[i] >= get_executor_num[i]:
                    job_valid[job_dag] = False
                    break
                else:
                    job_valid[job_dag] = True
            exec_level_idx_max=max(exec_level_idx.values())
            # print(type(exec_level_idx_max))
            # print(type(self.executor_levels))

            for l in range(exec_level_idx_max, len(self.executor_levels)):#应该考虑我们修改之后怎么编号，怎么屏蔽
                job_valid_mask[0, base + l] = 1

            base += self.executor_levels[-1]
        
        # print(job_valid)

        total_num_nodes = int(np.sum(
            job_dag.num_nodes for job_dag in job_dags))

        node_valid_mask = np.zeros([1, total_num_nodes])
        # print("frontier_nodes",len(frontier_nodes))

        # for node in frontier_nodes:
        #     #print(job_valid)
        #     if job_valid[node.job_dag]:
        #         act = action_map.inverse_map[node]
        #         node_valid_mask[0, act] = 1
                #print(node_valid_mask)
        for node in frontier_nodes:
            #print(job_valid)
            act = action_map.inverse_map[node]
            node_valid_mask[0, act] = 1


        return node_valid_mask, job_valid_mask, type_valid_mask

    def invoke_model(self, obs):
        # implement this module here for training
        # (to pick up state and action to record)
        node_inputs, job_inputs, \
            job_dags, \
            frontier_nodes, executor_limits, \
            exec_map, action_map,get_different_levels_executor_remain,get_executor_num = self.translate_state(obs)

        # get message passing path (with cache)
        gcn_mats, gcn_masks, dag_summ_backward_map, \
            running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags)

        # get node and job valid masks 增加了获得type
        node_valid_mask, job_valid_mask, type_valid_mask = \
            self.get_valid_masks(job_dags, frontier_nodes, exec_map, action_map,get_different_levels_executor_remain,get_executor_num)

        # get summarization path that ignores finished nodes
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        # invoke learning model
        node_act_probs, job_act_probs, type_act_probs, node_acts, job_acts, type_acts = \
            self.predict(node_inputs, job_inputs,
                node_valid_mask, job_valid_mask, type_valid_mask, \
                gcn_mats, gcn_masks, summ_mats, \
                running_dags_mat, dag_summ_backward_map)

        return node_acts, job_acts, type_acts, \
               node_act_probs, job_act_probs, type_act_probs, \
               node_inputs, job_inputs, \
               node_valid_mask, job_valid_mask, type_valid_mask,\
               gcn_mats, gcn_masks, summ_mats, \
               running_dags_mat, dag_summ_backward_map, \
               exec_map, job_dags_changed

    def get_action(self, obs): #加一个决策

        # parse observation
        job_dags, \
        frontier_nodes, executor_limits, \
         action_map,get_different_levels_executor_remain,get_executor_num = obs

        if len(frontier_nodes) == 0:
            # no action to take
            # return None, num_source_exec
            return None

        # invoking the learning model #这里加上了一些需要的参数
        node_act, job_act, type_act,\
            node_act_probs, job_act_probs, type_act_probs,\
            node_inputs, job_inputs, \
            node_valid_mask, job_valid_mask, type_valid_mask, \
            gcn_mats, gcn_masks, summ_mats, \
            running_dags_mat, dag_summ_backward_map, \
            exec_map, job_dags_changed = self.invoke_model(obs)

        if sum(node_valid_mask[0, :]) == 0:
            # no node is valid to assign
            # return None, num_source_exec
            return None

        # node_act should be valid
        #assert node_valid_mask[0, node_act[0]] == 1

        # parse node action
        node = action_map[node_act[0]]

        # find job index based on node
        job_idx = job_dags.index(node.job_dag)

        # job_act should be valid
        #assert job_valid_mask[0, job_act[0, job_idx] + \
            #len(self.executor_levels) * job_idx] == 1

        # find out the executor limit decision

        # if node.job_dag is source_job:
        #     agent_exec_act = self.executor_levels[
        #         job_act[0, job_idx]] - \
        #         exec_map[node.job_dag] + \
        #         num_source_exec
        # else:
        #     agent_exec_act = self.executor_levels[
        #         job_act[0, job_idx]] - exec_map[node.job_dag]



        use_type=type_act[0,job_idx]
        # agent_exec_act = self.executor_levels[
        #                      job_act[0, job_idx]] - executor_limits[node.job_dag][use_type]+1
        agent_exec_act = self.executor_levels[
                             job_act[0, job_idx]] - executor_limits[node.job_dag][use_type]

        # parse job limit action 这里的选择逻辑 在可分配和task和并行度间选一个

        # print("task",node.num_tasks - node.next_task_idx)
        # print("remain",get_different_levels_executor_remain[use_type])
        # print("exec_act",agent_exec_act)
        use_exec = min(
            len(node.remain_tasks),
            agent_exec_act, get_different_levels_executor_remain[use_type])



        assert node.mem <= self.exec_mem[use_type] and node.cpu <= self.type_num[use_type]

        return node, use_exec, use_type
