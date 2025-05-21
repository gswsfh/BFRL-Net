import itertools
import math
import random
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dim = 5
fea_dim= 16
action_dim = state_dim
hidden_dim = 32
entropy_rate = 0.1
reward_div_rate = 0.99
netUpdate_polyak = 0.995
buffer_size = 100000
printInfo = 10
epochs = 20
score_winsize = 16
startNetAction_step = 8 * score_winsize
sampleUpdateMin = fea_dim
sampleUpdateStep = 200
learning_rate_actor = 0.0001
learning_rate_critic = 0.0001
batch_size = 32

def loadRawData(filepath):
    data=np.load(filepath)
    return data

def constructDirectionDatasetFromMTWD(filepath):
    data=loadRawData(filepath)
    X=data["direction"]
    normalized_X = (X + 1) / 2
    y=data["label"]
    return {
        "X": normalized_X,
        "y": y
    }

def genScoreTable(dataset):
    table={i:list() for i in range(100)}
    X,y=dataset["X"],dataset["y"]
    datasetlen=len(y)
    for i in range(datasetlen):
        indices=np.where(y[i]==1)[0].tolist()
        for index in indices:
            table[index].append(X[i])




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            self.encoding[:, 1::2] = torch.cos(position * div_term[:])
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        expend = False
        if x.shape[-2] == 1:
            return x
        if len(x.shape) == 2:
            expend = True
            x = x.unsqueeze(0)
        totalFlowlen = x.size(-2) - 1
        splitEle = list()
        for i in range(totalFlowlen):
            x_f = torch.index_select(x, dim=-2, index=torch.tensor([i]).to(device))
            encode_f = x_f + self.encoding[:, 0, :].to(device)
            splitEle.append(encode_f)
        splitEle.append(
            torch.index_select(x, dim=-2, index=torch.tensor([totalFlowlen - 1]).to(device)).to(
                device) + torch.index_select(self.encoding,
                                             dim=-2,
                                             index=torch.tensor(
                                                 [1]).to(device)))
        x = torch.cat(splitEle, dim=-2)
        if expend:
            x = x.squeeze()
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=255):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        fea_d_model = fea_dim
        self.pos_encoder = PositionalEncoding(fea_d_model).to(device)
        self.activate=nn.ReLU()
        self.fn=nn.Linear(fea_d_model, hidden_dim)
        self.fn2=nn.Linear(hidden_dim, 1)
        self.initParams()

    def initParams(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0, std=1)

    def forward(self, state, deterministic=False):
        state = self.pos_encoder(state)
        state=self.activate(self.fn(state))
        state=self.fn2(state)
        actionRaw=state.squeeze(-1)
        action_softmax_p = torch.softmax(actionRaw, dim=-1)
        if deterministic:
            action_p, action_index = torch.max(action_softmax_p, dim=-1)
        else:
            action_index_pos = torch.multinomial(action_softmax_p, 1)
            action_index = action_index_pos.squeeze(-1)
            action_p = torch.gather(action_softmax_p, -1, action_index_pos).squeeze(-1)
        return action_index, action_p

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=255):
        super(CriticNet, self).__init__()
        self.ln=nn.Linear(state_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, 1)
        self.ln3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.initParams()

    def initParams(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0, std=0.01)

    def forward(self, state, action):
        rew = self.relu(self.ln(state))
        rew = self.relu(self.ln2(rew))
        rew=rew.squeeze(-1)
        action = action.unsqueeze(1)
        rew = torch.cat((rew, action), dim=-1)
        rew=self.ln3(rew)
        return rew

class ActorCritic(nn.Module):
    def __init__(self, fea_dim, act_dim, hidden_dim=hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = Actor(fea_dim, hidden_dim=hidden_dim)
        self.critic1 = CriticNet(fea_dim, hidden_dim=hidden_dim)
        self.critic2 = CriticNet(fea_dim, hidden_dim=hidden_dim)
    def act(self, state, deterministic=False):
        with torch.no_grad():
            actor, _ = self.actor(state, deterministic)
            return actor.cpu().numpy()

class ReplayBuffer:
    def __init__(self, state_dim,fea_dim, size):
        self.state_buf = np.zeros((size, state_dim,fea_dim), dtype=np.float32)
        self.state2_buf = np.zeros((size, state_dim,fea_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.maxsize = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = state
        self.state2_buf[self.ptr] = next_state
        self.rew_buf[self.ptr] = reward
        self.act_buf[self.ptr] = action
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.maxsize
        self.size = min(self.size + 1, self.maxsize)

    def sample_batch(self, batch_size):
        idxs = np.arange(self.size + self.maxsize, self.size + self.maxsize - batch_size, -1) % self.maxsize

        batch = dict(state=self.state_buf[idxs],
                     state_next=self.state2_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def reset(self):
        self.state_buf.fill(0)
        self.state2_buf.fill(0)
        self.act_buf.fill(0)
        self.rew_buf.fill(0)
        self.done_buf.fill(0)
        self.ptr, self.size = 0, 0

class SAC:
    def __init__(self, dataset, state_dim, fea_dim, act_dim, hidden_dim=32):
        self.dataset = dataset
        self.datasetLen = len(dataset["y"])
        self.traffic_score_table = dict()
        self.traffic_score_ids_table = dict()
        self.calTrafficScore()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.actor_critic = ActorCritic(fea_dim=fea_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
        self.actor_critic_target = deepcopy(self.actor_critic)
        self.freezeTargetAC()
        self.critic_params = itertools.chain(self.actor_critic.critic1.parameters(),
                                             self.actor_critic.critic2.parameters())
        self.replay_buffer = ReplayBuffer(state_dim=state_dim,fea_dim=fea_dim, size=buffer_size)
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = Adam(self.critic_params, lr=learning_rate_critic)

    def freezeTargetAC(self):
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

    def setSeed(self, seed=20240511):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def countVars(self, module):
        return sum([np.prod(p.shape) for p in module.parameters()])

    def getTotalVars(self):
        return tuple(
            self.countVars(module) for module in
            [self.actor_critic.actor, self.actor_critic.critic1, self.actor_critic.critic2])

    def calTrafficScore(self, winsize=score_winsize,websize=100):
        self.traffic_score_table={i:dict() for i in range(websize)}
        for i in range(self.datasetLen):
            X_i=self.dataset["X"][i]
            y_i=self.dataset["y"][i]
            groupIndex = np.where(y_i == 1)[0].tolist()
            for j in groupIndex:
                for i in range(0, 1, len(X_i) - winsize):
                    i_group = X_i[i:i + winsize].tolist()
                    i_group=tuple(i_group)
                    if i_group not in self.traffic_score_table[j]:
                        self.traffic_score_table[j][i_group] = 1
                    else:
                        self.traffic_score_table[j][i_group] += 1

    def calLossCritic(self, data):
        state, action_index, reward, state_next, done = data["state"].to(device), torch.as_tensor(data["action"]).to(
            device), data["reward"].to(device), \
            data[
                "state_next"].to(device), \
            data["done"].to(device),
        action_index = action_index.to(torch.long)
        q1 = self.actor_critic.critic1(state,action_index)
        q2 = self.actor_critic.critic2(state,action_index)
        with torch.no_grad():
            action_next_index, action_p_next = self.actor_critic.actor(state_next)
            q1_pi_targ = self.actor_critic_target.critic1(state_next, action_next_index)
            q2_pi_targ = self.actor_critic_target.critic2(state_next, action_next_index)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            bellman_value = reward + reward_div_rate * (1 - done) * (q_pi_targ - entropy_rate * action_p_next)

        loss_q1 = ((q1 - bellman_value) ** 2).mean()
        loss_q2 = ((q2 - bellman_value) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def get_action(self, state, deterministic=False):
        return self.actor_critic.act(torch.as_tensor(state, dtype=torch.float32).to(device),
                                     deterministic=deterministic)

    def getRandomAction(self, state):
        return random.choice([i for i in range(state.shape[-2])])

    def act(self, traffic_cls, actionindex, data):
        traffic_cls_next=traffic_cls.copy()
        cur_line=traffic_cls_next[actionindex]
        cur_line= np.delete(cur_line, 0)
        cur_line = np.append(cur_line, data)
        traffic_cls_next[actionindex] = cur_line
        return traffic_cls_next

    def groupAct(self, traffic_group, traffic_group_infos, actionindex, data):
        actionindex = int(actionindex)
        if actionindex == len(traffic_group):
            traffic_group.append([data[1]["ids"]])
            traffic_group_infos.append([(data[0], data[1]["rawdata"], data[2])])
        else:
            traffic_group[actionindex].append(data[1]["ids"])
            traffic_group_infos[actionindex].append((data[0], data[1]["rawdata"], data[2]))

    def optimized_match_sequences(self,pair, center):
        i, j = 0, 0
        match_count = 0
        n, m = len(pair), len(center)
        while j < m:
            if i >= n:
                break
            if pair[i] == center[j]:
                match_count += 1
                i += 1
                j += 1
            else:
                j += 1
                if j >= m:
                    break
        return match_count

    def calActReward(self, traffic_cls, actionindex,catagory):

        def judgeScore(feas,catagory):
            if 0.25 in feas:
                non_index=0
                for item_i,item in enumerate(feas):
                    if item!=0.25:
                        non_index=item_i
                        break
                feas=feas[non_index:]
            feas = tuple(feas)
            if len(feas)<0.5*score_winsize:
                return 0
            for k,v in self.traffic_score_table[catagory].items():
                pairnums=self.optimized_match_sequences(feas,k)
                if pairnums >= 0.5*score_winsize:
                    return v
            return -1

        assert traffic_cls.shape[-1] == fea_dim

        return judgeScore(traffic_cls[actionindex], catagory)

    def calGroupsScore(self, groups):
        score = 0
        for group in groups:
            if len(group) == 1:
                pass
            else:
                for i in range(len(group) - 1):
                    key = str(group[i] + group[i + 1])
                    if key in self.traffic_score_table:
                        score += self.traffic_score_table[key]
        return score

    def test_agent(self,dataset_id):
        test_groups = list()
        test_groups_infos = list()
        X=self.dataset["X"][dataset_id]
        y=self.dataset["y"][dataset_id]
        y_i_group_indexs = np.where(y == 1)[0]
        y_i_group_len = len(y_i_group_indexs)
        y_i_group_indexs_table = {i: item for i, item in enumerate(y_i_group_indexs)}
        test_state=None
        cls_state_addflow=None
        reward_total=0
        for pkt in X:
            if test_state is None:
                test_state = np.zeros((y_i_group_len, fea_dim), dtype=np.float32)
                cls_state_addflow = self.stateAddFlow(test_state, pkt)
            action = self.get_action(cls_state_addflow, deterministic=True)
            action=int(action)
            cls_state_next = self.act(test_state, action, pkt)
            test_state=cls_state_next
            action_catagory = y_i_group_indexs_table[action]
            reward = self.calActReward(cls_state_next, action, action_catagory)
            reward_total+=reward
        return reward_total

    def predict_agent(self,dataset_id):
        test_groups = list()
        test_groups_infos = list()
        X = self.dataset["X"][dataset_id]
        y = self.dataset["y"][dataset_id]
        y_i_group_indexs = np.where(y == 1)[0]
        y_i_group_len = len(y_i_group_indexs)
        y_i_group_indexs_table = {i: item for i, item in enumerate(y_i_group_indexs)}
        test_state = None
        cls_state_addflow = None
        reward_total = 0
        res=None
        for pkt in X[:1000]:
            if test_state is None:
                test_state = np.zeros((y_i_group_len, fea_dim), dtype=np.float32)
                test_state.fill(0.5)
                res=test_state.copy().tolist()
            cls_state_addflow = self.stateAddFlow(test_state, pkt)
            action = self.get_action(cls_state_addflow, deterministic=True)
            action = int(action)
            cls_state_next = self.act(test_state, action, pkt)
            test_state = cls_state_next
            res[action].append(int(pkt))
        res=[item[fea_dim:] for item in res]
        return res

    def calLossActor(self, data):
        state = data["state"].to(device)
        action_index, action_p = self.actor_critic.actor(state)
        q1_pi = self.actor_critic.critic1(state, action_index)
        q2_pi = self.actor_critic.critic2(state, action_index)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (entropy_rate * action_p - q_pi).mean()
        return loss_pi

    def update(self, data):
        res = {
            "act_loss": 0,
            "critic_loss": 0
        }
        self.critic_optimizer.zero_grad()
        loss_critic = self.calLossCritic(data)
        res["critic_loss"] = loss_critic.cpu().detach().numpy()
        loss_critic.backward()
        self.critic_optimizer.step()
        for p in self.critic_params:
            p.requires_grad = False
        self.actor_optimizer.zero_grad()
        loss_action = self.calLossActor(data)
        res["act_loss"] = loss_action.cpu().detach().numpy()
        loss_action.backward()
        self.actor_optimizer.step()
        for p in self.critic_params:
            p.requires_grad = True
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                p_targ.data.mul_(netUpdate_polyak)
                p_targ.data.add_((1 - netUpdate_polyak) * p.data)
        return res

    def stateAddFlow(self,state,pkt):
        cls_state_addflow = state[:, 1:]
        cls_state_addflow = np.hstack((cls_state_addflow, np.full((cls_state_addflow.shape[0], 1), pkt)))
        return cls_state_addflow

    def run(self, modelpath):
        print("start training")
        select_flow_catagory=49
        for epoch in range(epochs):
            select_flow_num = 10
            print(f"Epoch：{epoch}/{epochs}")
            for dataset_i in range(1000):
                y_i=self.dataset["y"][dataset_i]
                y_i_group_indexs=np.where(y_i==1)[0]
                if select_flow_catagory not in y_i_group_indexs:
                    continue
                select_flow_num -= 1
                if select_flow_num<0:
                    continue
                X_i = self.dataset["X"][dataset_i]
                y_i_group_len=len(y_i_group_indexs)
                y_i_group_indexs_table={i:item for i,item in enumerate(y_i_group_indexs)}
                X_i_len=len(X_i)
                cls_state = None
                cls_state_addflow=None
                self.replay_buffer.reset()
                print(f"Flow:{dataset_i}/{self.datasetLen}")
                for packet_i in range(X_i_len):
                    cur_packet=X_i[packet_i]
                    if packet_i == X_i_len - 1:
                        done = True
                    else:
                        done = False
                    if cls_state is None:
                        cls_state = np.zeros((y_i_group_len, fea_dim),dtype=np.float32)
                        cls_state.fill(0.5)
                    cls_state_addflow=self.stateAddFlow(cls_state, cur_packet)
                    if packet_i > startNetAction_step:
                        action = self.get_action(cls_state_addflow, deterministic=False)
                        action=int(action)
                    else:
                        action = self.getRandomAction(cls_state_addflow)
                    cls_state_next = self.act(cls_state, action, cur_packet)
                    if dataset_i+1<X_i_len:
                        packet_i_1=X_i[dataset_i+1]
                        cls_state_next_addflow=self.stateAddFlow(cls_state_next, packet_i_1)
                    else:
                        cls_state_next_addflow = self.stateAddFlow(cls_state_next, 0)
                    action_catagory=y_i_group_indexs_table[action]
                    reward = self.calActReward(cls_state_next, action,action_catagory)
                    self.replay_buffer.store(cls_state_addflow, action, reward, cls_state_next_addflow, done)
                    cls_state = cls_state_next
                    if packet_i % sampleUpdateStep == 0 and packet_i >= sampleUpdateMin:
                        updatestep = int(packet_i / sampleUpdateStep)
                        loss = [list(), list()]
                        # for _ in range(sampleUpdateStep):
                        #     batch = self.replay_buffer.sample_batch(batch_size=batch_size)
                        #     update_info = self.update(batch)
                        #     loss[0].append(update_info["act_loss"])
                        #     loss[1].append(update_info["critic_loss"])
                        for _ in range(batch_size):
                            batch = self.replay_buffer.sample_batch(batch_size=sampleUpdateStep)
                            update_info = self.update(batch)
                            loss[0].append(update_info["act_loss"])
                            loss[1].append(update_info["critic_loss"])
                        if updatestep % printInfo == 0:
                            print(f"Updated! actor_loss: {np.mean(loss[0])} , critic_loss: {np.mean(loss[1])}")

                torch.save(self.actor_critic.state_dict(), modelpath)



if __name__ == '__main__':

    filepath = "filepath"
    dataset = constructDirectionDatasetFromMTWD(filepath)
    sac = SAC(dataset, state_dim=action_dim, act_dim=action_dim, fea_dim=fea_dim)
    modelpath = "./model.pt"

    step = 0
    if step == 0:
        retrain=True
        if not retrain:
            sac.actor_critic.load_state_dict(torch.load(modelpath))
        sac.run(modelpath)


    elif step == 1:
        sac.actor_critic.load_state_dict(torch.load(modelpath))
        select_flow_catagory=49
        X = dataset["X"]
        y = dataset["y"]
        datasetlen = len(y)
        select_flow_ids=list()
        select_num=10
        for i in range(datasetlen):
            y_i = y[i]
            x_i = X[i]
            y_i_group_indexs = np.where(y_i == 1)[0]
            if select_flow_catagory in y_i_group_indexs:
                select_flow_ids.append([i,y_i_group_indexs])
                select_num-=1
                if select_num==0:
                    break
        for item in select_flow_ids:
            index=item[0]
            select_index = np.where(item[1] == select_flow_catagory)[0][0]
            res = sac.predict_agent(index)
            print(res[select_index])


