

import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        #当列表内元素大于max_size，另一端的数据会减去
        self.buffer_normal = collections.deque(maxlen=max_size)
        good_experience_size = int(max_size / 100)
        self.buffer_good = collections.deque(maxlen=good_experience_size)

    def append_normal(self, exp):
        self.buffer_normal.append(exp)

    def append_god(self, exp):
        self.buffer_good.append(exp)
        
        s, a, r, s_p, done = exp
        u = np.argmax(a)
        a2 = np.zeros(4)
        a3 = np.zeros(4)
        a4 = np.zeros(4)

        
        x1 = s[0]
        x2 = s[1]
        x3 = s_p[0]
        x4 = s_p[1]

        s_2 = [-x2,x1]
        s_p_2 = [-x4,x3]
        if(u == 0):
            a2[2] = 1
        elif(u == 2):
            a2[1] = 1
        elif(u == 1):
            a2[3] = 1
        elif(u == 3):
            a2[0] = 1
        exp2 = (s_2,a2,r,s_p_2,done)
        self.buffer_good.append(exp2)

        s_3 = [-x1, -x2]
        s_p_3 = [-x3, -x4]
        if(u == 0):
            a3[1] = 1
        elif(u == 1):
            a3[0] = 1
        elif(u == 2):
            a3[3] = 1
        elif(u == 3):
            a3[2] = 1
        exp3 = (s_3,a3,r,s_p_3,done)
        self.buffer_good.append(exp3)

        s_4 = [-x1, -x2]
        s_p_4 = [-x3, -x4]
        if(u == 0):
            a4[3] = 1
        elif(u == 3):
            a4[1] = 1
        elif(u == 1):
            a4[2] = 1
        elif(u == 2):
            a4[0] = 1
        exp4 = (s_4,a4,r,s_p_4,done)
        self.buffer_good.append(exp4)
        

    def sample(self, batch_size):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        if(self.len_of_good() > 0 and 0):
            mini_batch1 = random.sample(self.buffer_good, 1)
            for experience in mini_batch1:
                s, a, r, s_p, done = experience
                obs_batch.append(s)
                action_batch.append(a)
                reward_batch.append(r)
                next_obs_batch.append(s_p)
                done_batch.append(done)

            mini_batch2 = random.sample(self.buffer_normal, batch_size - 1)
            for experience in mini_batch2:
                s, a, r, s_p, done = experience
                obs_batch.append(s)
                action_batch.append(a)
                reward_batch.append(r)
                next_obs_batch.append(s_p)
                done_batch.append(done)
        else:
            mini_batch = random.sample(self.buffer_normal, batch_size)
            for experience in mini_batch:
                s, a, r, s_p, done = experience
                obs_batch.append(s)
                action_batch.append(a)
                reward_batch.append(r)
                next_obs_batch.append(s_p)
                done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer_normal)

    def len_of_good(self):
        return len(self.buffer_good)
