import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Individual:
    def __init__(self, pop, idx):
        self.pop = pop  # 引用 Population 对象
        self.idx = idx  # 个体索引

    @property
    def dec(self):
        # 动态获取 Population 的 decs 数组中的对应行
        return self.pop.decs[self.idx]

    @dec.setter
    def dec(self, value):
        # 动态设置 Population 的 decs 数组中的对应行
        self.pop.decs[self.idx] = value

    @property
    def obj(self):
        return self.pop.objs[self.idx]

    @obj.setter
    def obj(self, value):
        self.pop.objs[self.idx] = value

    @property
    def cst(self):
        return self.pop.csts[self.idx]

    @cst.setter
    def cst(self, value):
        self.pop.csts[self.idx] = value


class Population:
    def __init__(self,decs=None,objs=None,csts=None,lb=None,ub = None):
        self.lb = lb
        self.ub = ub
        self.decs=decs # ndim = 2, shape = (popSize,decDim)
        self.objs=objs.reshape(self.decs.shape[0],) if objs.ndim==1 else objs# ndim = 2, shape = (popSize,objDim)
        self.csts=csts # ndim = 2, shape = (popSize,cstNum)
        self.inds = [Individual(self,idx=i) for i in range(self.decs.shape[0])]

        self.cstNum = None # ndim = 0, scalar 
        self.cstViolationFlag = None # ndim = 2, shape = (popSize,cstNum) ,True  violated
        self.cstViolationNum  = None # ndim = 1, shape = (popSize,)
        if csts is not None:
            self.Cal_Constrain()
        else:
            self.feasible = [True]*self.decs.shape[0]

    def __getitem__(self, idx):
        return self.inds[idx]
    # def addInd(self,dec,obj,cst=None):
    #     self.inds.append(ind)
    #     self.decs = np.vstack((self.decs,dec))
    #     self.objs = np.append(self.objs,obj)
    #     self.csts = np.vstack((self.csts,cst))
    #     self.feasible = np.append(self.feasible,ind.feasible)
    def Cal_Constrain(self):
            self.cstNum = self.csts.shape[1]
            # 计算约束违反情况
            cstMax = self.csts.max(axis=0)
            self.cstViolationFlag = np.zeros_like(self.csts)
            self.cstViolationNum = np.zeros(self.csts.shape[0])
            self.feasible= np.zeros(self.csts.shape[0])
            for i in range (self.csts.shape[0]):
                self.cstViolationFlag[i,:] = np.array([False if con < 0 else True for con in self.csts[i]])
                self.cstViolationNum[i] = np.sum(self.cstViolationFlag[i] == True)
                self.feasible[i] = True if self.cstViolationNum[i] == 0 else False
                if not self.feasible[i]:
                    self.csts[i,self.csts[i] > 0] /= cstMax[self.csts[i] > 0]
                    self.p = (1+np.max(self.csts[i]))*(1+self.cstViolationNum[i]/self.cstNum)
    @property
    def popSize(self):
        return len(self.inds)
    @property
    def bestInd(self):
        return self.inds[np.argmin(self.objs[self.feasible])]


class RLMSO:
    def __init__(self, Threshold=0.25,Thresold2= 0.6,C1=0.5,C2=0.05,C3=2):
        self.Threshold=Threshold
        self.Thresold2=Thresold2
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.Nm =None
        self.Nf =None
    def divide_swarm(self, X_dec, fitness,csts=None):
        self.Nm = round(len(X_dec) / 2)  # eq.(2&3)
        self.Nf = len(X_dec) - self.Nm
        Xm = Population(decs=X_dec[:self.Nm, :], objs=fitness[:self.Nm],
                        csts=self.csts[:self.Nm] if csts is not None else None)
        Xf = Population(decs=X_dec[self.Nm:, :], objs=fitness[self.Nm:],
                        csts=self.csts[self.Nm:] if csts is not None else None)
        Xfood = Xm.bestInd if Xm.bestInd.obj < Xf.bestInd.obj else Xf.bestInd
        return Xm, Xf, Xfood
    
    def init_pop(self,N,lb,ub,dim):
        # 确保 lb 和 ub 是 numpy 数组
        lb = np.array(lb, ndmin=1)  # 转换为至少 1D 数组
        ub = np.array(ub, ndmin=1)  # 转换为至少 1D 数组
        # 如果 lb 和 ub 是标量，则扩展它们到 dim 的大小
        if lb.size == 1:
            lb = np.full((dim,), lb)
        if ub.size == 1:
            ub = np.full((dim,), ub)
        X = lb + np.random.rand(N, dim) * (ub - lb)  # eq.(1)
        return X,lb,ub
    
    def ind_exploration_NoFood(self,X_dec, fitness, X_decs,fitnesses, C2, lb, ub):
        vec_flag = np.array([1, -1])  # Equivalent to MATLAB's [1, -1]
        N,dim = X_decs.shape
        newX_dec = np.zeros(dim)  # 1D array for new decision variables
        for j in range(dim):
            rand_leader_index = int(N * np.random.rand())  # Random index in range [0, N-1]
            X_rand = X_decs[rand_leader_index, :]
            flag_index = int(2 * np.random.rand())  # Random index in range [0, 1]
            Flag = vec_flag[flag_index]
            A = np.exp(-fitnesses[rand_leader_index] / (fitness + np.finfo(float).eps))  # Avoid division by zero
            newX_dec[j] = X_rand[j] + Flag * C2 * A * ((ub[j] - lb[j]) * np.random.rand() + lb[j])

        return newX_dec
  

    def exploration_NoFood(self,X_dec, fitness, C2, lb, ub):
        vec_flag = np.array([1, -1])  # Equivalent to MATLAB's [1, -1]
        N, dim = X_dec.shape
        newX_dec = np.zeros((N, dim))  # Initialize new decision variables matrix

        for i in range(N):
            for j in range(dim):
                rand_leader_index = int(N * np.random.rand())  # Random index in range [0, N-1]
                X_rand = X_dec[rand_leader_index, :]
                flag_index = int(2 * np.random.rand())  # Random index in range [0, 1]
                Flag = vec_flag[flag_index]
                A = np.exp(-fitness[rand_leader_index] / (fitness[i] + np.finfo(float).eps))  # Avoid division by zero
                newX_dec[i, j] = X_rand[j] + Flag * C2 * A * ((ub[j] - lb[j]) * np.random.rand() + lb[j])

        return newX_dec
    import numpy as np

    def exploit_Food(self,X_dec, Xfood, Temp, C3):
        vec_flag = np.array([1, -1])  # Equivalent to MATLAB's [1, -1]
        N, dim = X_dec.shape
        newX_dec = np.zeros((N, dim))  # Initialize the new decision variables matrix

        for i in range(N):
            flag_index = int(2 * np.random.rand())  # Random index in range [0, 1]
            Flag = vec_flag[flag_index]
            for j in range(dim):
                newX_dec[i, j] = Xfood[j] + C3 * Flag * Temp * np.random.rand() * (Xfood[j] - X_dec[i, j])  # eq.(7)

        return newX_dec
    def so_fight(self,X_dec, fitness, Xbest_dec, fitnessBest, t1, C3, Q):
        N, dim = X_dec.shape
        newX_dec = np.zeros((N, dim))  # Initialize the new decision variables matrix

        for i in range(N):
            for j in range(dim):
                F = np.exp(-fitnessBest / (fitness[i] + np.finfo(float).eps))  # Avoid division by zero
                newX_dec[i, j] = t1 * X_dec[i, j] + C3 * F * np.random.rand() * (Q * Xbest_dec[j] - X_dec[i, j])  # eq.(8)

        return newX_dec
    def so_mating(self,Xm_dec, Xf_dec, fitness_m, fitness_f, C3, Q, lb, ub):
        Nm, dim = Xm_dec.shape
        newXm_dec = np.zeros((Nm, dim))  # Initialize updated male decision matrix

        for i in range(Nm):
            for j in range(dim):
                Mm = np.exp(-fitness_f[i] / (fitness_m[i] + np.finfo(float).eps))  # Avoid division by zero
                newXm_dec[i, j] = Xm_dec[i, j] + C3 * np.random.rand() * Mm * (Q * Xf_dec[i, j] - Xm_dec[i, j])  # eq.(10)

        Nf, dim = Xf_dec.shape
        newXf_dec = np.zeros((Nf, dim))  # Initialize updated female decision matrix

        for i in range(Nf):
            for j in range(dim):
                Mf = np.exp(-fitness_m[i] / (fitness_f[i] + np.finfo(float).eps))  # Avoid division by zero
                newXf_dec[i, j] = Xf_dec[i, j] + C3 * np.random.rand() * Mf * (Q * Xm_dec[i, j] - Xf_dec[i, j])  # eq.(11)
        return newXm_dec, newXf_dec
    
    import numpy as np

    def cal_diversity(self,X_dec):
        N, dim = X_dec.shape
        d_pop = 0
        diversity = np.zeros(N)
        x_mean = np.mean(X_dec, axis=0)  # Mean across individuals

        for i in range(N):
            d_ind = 0
            for j in range(dim):
                d_ind += (X_dec[i, j] - x_mean[j]) ** 2
            d_pop += d_ind
            diversity[i] = np.sqrt(d_pop)

        return diversity


    def get_neighbor_diversity(self,X, k):
        N, dim = X.shape

        # Find nearest neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        nearest_indices_all = nbrs.kneighbors(X, return_distance=False)

        neighbor_diversities = np.zeros(N)

        for i in range(N):
            neighbor_dec = X[nearest_indices_all[i], :]  # Extract neighbor decision variables
            ub2 = np.max(neighbor_dec, axis=0)  # Upper bounds for neighbors
            lb2 = np.min(neighbor_dec, axis=0)  # Lower bounds for neighbors
            Diagonal_Length = np.sqrt(np.sum((ub2 - lb2) ** 2))
            diversity_values = self.cal_diversity(neighbor_dec)  # Calculate diversity for neighbors
            neighbor_diversities[i] = np.sum(diversity_values) / ((k + 1) * (Diagonal_Length + 1e-160))

        return neighbor_diversities
    def get_state_knn(self,X, fitness, k):
        N, _ = X.shape
        state = np.zeros((N, 2), dtype=int)

        # Calculate neighbor diversity
        D = self.get_neighbor_diversity(X, k)

        # Calculate population diversity and diagonal length
        ub2 = np.max(X, axis=0)
        lb2 = np.min(X, axis=0)
        DL = np.sqrt(np.sum((ub2 - lb2) ** 2))
        popD = np.sum(self.cal_diversity(X)) / (N * (DL + 1e-160))
        popF = np.mean(fitness)

        # Normalize diversity (RDs) and fitness (RFs)
        if popD == 0:
            RDs = np.zeros_like(D)
        else:
            RDs = D / popD

        if popF == 0:
            RFs = np.zeros_like(fitness)
        else:
            RFs = fitness / popF

        # Scale RDs and RFs to [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        RDs = scaler.fit_transform(RDs.reshape(-1, 1)).flatten()
        RFs = scaler.fit_transform(RFs.reshape(-1, 1)).flatten()

        state[RDs < 0.2, 1] = 1
        state[(RDs >= 0.2) & (RDs < 0.4), 1] = 2
        state[(RDs >= 0.4) & (RDs < 0.6), 1] = 3
        state[(RDs >= 0.6) & (RDs < 0.8), 1] = 4
        state[RDs >= 0.8, 1] = 0

        state[RFs < 0.2, 0] = 1
        state[(RFs >= 0.2) & (RFs < 0.4), 0] = 2
        state[(RFs >= 0.4) & (RFs < 0.6), 0] = 3
        state[(RFs >= 0.6) & (RFs <= 0.8), 0] = 4
        state[RFs > 0.8, 0] = 0


        return state
    
    import numpy as np

    def update_q_table(self,s, a, r, s_next, q, alpha=0.1, gamma=0.9):
        actions = q[s_next[0], s_next[1], :]
        q_target_value = np.max(actions)
        
        # Update Q value for current state-action pair
        q[s[0], s[1], a] += alpha * (r + gamma * q_target_value - q[s[0], s[1], a])

        return q
    def get_action(self,q_table, state):
        N = state.shape[0]
        action = np.zeros(N, dtype=int)
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        for i in range(N):
            actions_value = q_table[state[i, 0], state[i, 1], :]
            probability = softmax(actions_value)
            action[i] = np.random.choice(len(probability), p=probability)

        return action

    def evaluation_reward(self,X, newX_dec, lb, ub, fobj, failure_times):
        N = X.decs.shape[0]  # 获取个体数量
        fitness_new = np.zeros(X.objs.shape)  # 新的适应度
        fitness_old = X.objs  # 旧的适应度
        reward = -1 * np.ones(N)  # 初始奖励为 -1
        newX_dec = np.clip(newX_dec, lb, ub)
        X_dec = X.decs
        t1 = time.time()
        def compute_fitness(j, newX_dec, fobj):
            print(j)
            return fobj(newX_dec[j, :])
        # fitness_new = Parallel(n_jobs=25)(delayed(compute_fitness)(j, newX_dec, fobj) for j in range(N))

        # for j in range(N):
        #     # 计算新的适应度
        #     fitness_new[j] = fobj(newX_dec[j, :])
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_fitness, j, newX_dec, fobj) for j in range(N)]
            for j, future in enumerate(futures):
                fitness_new[j] = future.result()

        t2 = time.time()
        print(t2-t1)
        # fitness_new = parallel_computation(N, newX_dec, fobj)
        improved_mask = fitness_new < fitness_old

        # 更新 X_dec 中适应度提高的行
        X_dec[improved_mask, :] = newX_dec[improved_mask, :]
        fitness_old[improved_mask] = fitness_new[improved_mask]
        reward[improved_mask] = 1

        # 更新 failure_times，对于未提高的增加 1
        failure_times[improved_mask] = 0
        failure_times[~improved_mask] += 1

        newX = Population(decs=X_dec, objs=fitness_old)
        return newX, reward, failure_times
    def init_Q_parameter(self):
        RF_num = 5
        RD_num = 5
        strategy_num = 4
        q_table_m = np.zeros((RF_num, RD_num, strategy_num))

        return q_table_m
    def updateXbest(self,Xm, Xf, fitness_m, fitness_f, Xbest_m, Xbest_f, fitnessBest_m, fitnessBest_f):
        # 找到 Xm 和 Xf 中最小适应度值的索引
        Ybest1, gbest1 = np.min(fitness_m), np.argmin(fitness_m)
        Ybest2, gbest2 = np.min(fitness_f), np.argmin(fitness_f)
        
        # 更新 Xbest_m 和 fitnessBest_m
        if Ybest1 < fitnessBest_m:
            Xbest_m = Xm[gbest1, :]
            fitnessBest_m = Ybest1
        
        # 更新 Xbest_f 和 fitnessBest_f
        if Ybest2 < fitnessBest_f:
            Xbest_f = Xf[gbest2, :]
            fitnessBest_f = Ybest2
        
        # 根据 fitnessBest_m 和 fitnessBest_f 来更新 GYbest 和 Xfood
        if fitnessBest_m < fitnessBest_f:
            GYbest = fitnessBest_m
            Xfood = Xbest_m
        else:
            GYbest = fitnessBest_f
            Xfood = Xbest_f
        
        return Xbest_m, Xbest_f, fitnessBest_m, fitnessBest_f, GYbest, Xfood
    @staticmethod
    def sigmoid(x):
        """计算 Sigmoid 函数值"""
        return 1 / (1 + np.exp(-x))
    
    def binary(self, X):
        """将 X 转换为 Binary 表示"""
        # 确保输入是 NumPy 数组
        X = np.array(X)
        
        # 获取与 X 相同形状的随机数矩阵
        rand_matrix = np.random.rand(*X.shape)
        
        # 计算 Sigmoid(X)
        sigmoid_X = self.sigmoid(X)
        
        # 根据 rand < sigmoid_X 生成 binary_X
        binary_X = (rand_matrix < sigmoid_X).astype(int)
        
        return binary_X

    def evaluate(self,X_dec, fobj,):
        N = X_dec.shape[0]
        fitness = np.zeros(N)
        for i in range(N):
            fitness[i] = fobj(X_dec[i, :])

        return fitness


    def optimize(self,N,T,lb,ub,dim,fobj):
        Threshold = self.Threshold
        Threshold2 = self.Thresold2
        C1 = self.C1 * np.ones(T)
        C2 = self.C2 * np.ones(T)
        C3 = self.C3 * np.ones(T)
        t1 = np.ones(T)
        ## Init
        X_dec,lb,ub= self.init_pop(N,lb,ub,dim)
        fitness = np.zeros(N)
        for i in range(N):
            fitness[i] = fobj(X_dec[i, :])
        gbest_t = np.zeros(T)

        Xm, Xf, Xfood = self.divide_swarm(X_dec, fitness)
        q_table_m = self.init_Q_parameter()
        
        X_out = []
        fitness_out = []

        failure_times_m = np.zeros(Xm.popSize)  # Assuming Nm is defined
        failure_times_f = np.zeros(Xf.popSize)  # Assuming Nf is defined

        maxFailure_times = 10

        for t in range(1, T ):
            Temp = np.exp(-t / T)  # eq.(4)
            Q_val = C1[t - 1] * np.exp((t - T) / T)  # eq.(5)

            # 找到超过最大失败次数的个体
            indices = np.where(failure_times_m >= maxFailure_times)[0].ravel()
            indices1 = np.where(failure_times_f >= maxFailure_times)[0].ravel()
            if len(indices) > 0 or len(indices1) > 0:
                if len(indices) > 0:
                    
                    X_out = np.vstack([X_out, Xm.decs[indices, :]]) if len(X_out) > 0 else Xm.decs[indices, :]
                    fitness_out = np.hstack([fitness_out, Xm.objs[indices]]) if len(fitness_out) > 0 else Xm.objs[indices]

                if len(indices1) > 0:
                    X_out = np.vstack([X_out, Xf.decs[indices1, :]]) if len(X_out) > 0 else Xf.decs[indices1, :]
                    fitness_out = np.hstack([fitness_out, Xf.objs[indices1]]) if len(fitness_out) > 0 else Xf.objs[indices1]
                a = Xm.decs[indices, :]
                b = Xm.decs[indices1, :]
                sortedIndex = np.argsort(fitness_out)
                X_out = X_out[sortedIndex, :]
                fitness_out = fitness_out[sortedIndex]

                if len(fitness_out) > N:
                    X_out = X_out[:N, :]
                    fitness_out = fitness_out[:N]

                mF_max = 0.9
                mF_min = 0.1
                mF = mF_max - (mF_max - mF_min) * (t / T)
                F = (mF - 0.1) + 0.2 * np.random.rand()  # 生成随机数F

                p = round(0.05 * len(fitness_out))
                p = max(p, 1)  # 确保 p 至少为 1

                xall = np.vstack([X_out, Xm.decs, Xf.decs])
                fitnesspbest = np.hstack([fitness_out, Xm.objs, Xf.objs])

                pIndex = np.argsort(fitnesspbest)
                xpbest = xall[pIndex[:p], :]

                for i in range(len(indices)):
                    r = np.random.choice(xall.shape[0], 2, replace=False)
                    x_selected = xall[r, :]
                    randp = np.random.randint(0, p)
                    tempXm = xpbest[randp, :] + F * (x_selected[0, :] - x_selected[1, :])

                    # 边界检查
                    tempXm = np.clip(tempXm, lb, ub)
                    y = fobj(tempXm)
                    Xm.objs[indices[i]] = y
                    Xm.decs[indices[i], :] = tempXm

                for i in range(len(indices1)):
                    r = np.random.choice(xall.shape[0], 2, replace=False)
                    x_selected = xall[r, :]
                    randp = np.random.randint(0, p)
                    tempXf = xpbest[randp, :] + F * (x_selected[0, :] - x_selected[1, :])

                    # 边界检查
                    tempXf = np.clip(tempXf, lb, ub)
                    y = fobj(tempXf)
                    Xf.objs[indices1[i]] = y
                    Xf.decs[indices1[i], :] = tempXf

            # 获取 Q-learning 状态和行动
            state_m = self.get_state_knn(Xm.decs,Xm.objs, round(N/10))
            action_m = self.get_action(q_table_m, state_m)
            newXm_dec = np.zeros_like(Xm.decs)
            
            for i in range(Xm.decs.shape[0]):
                if action_m[i] == 1:
                    newXm_dec[i, :] = self.ind_exploration_NoFood(Xm.decs[i],Xm.objs[i],Xm.decs,Xm.objs, C2[t - 1], lb, ub)
                elif action_m[i] == 2:
                    newXm_dec[i, :] = self.exploit_Food(Xm.decs[i].reshape(1, -1), Xfood.dec, Temp, C3[t - 1])
                elif action_m[i] == 3:
                    newXm_dec[i, :] = self.so_fight(Xm.decs[i].reshape(1, -1), Xm.objs[i].reshape(1, ), Xf.bestInd.dec, Xf.bestInd.obj, t1[t - 1], C3[t - 1], Q_val)
                else:
                    newXm_dec[i, :], _ = self.so_mating(Xm.decs[i].reshape(1, -1), Xf.decs[i].reshape(1, -1), Xm.objs[i].reshape(1, ), Xf.objs[i].reshape(1, ), C3[t - 1], Q_val, lb, ub)

            # 更新 Xf 解
            if Q_val < Threshold:
                newXf_dec = self.exploration_NoFood(Xf.decs, Xf.objs, C2[t - 1], lb, ub)
            else:
                if Temp > Threshold2:
                    newXf_dec = self.exploit_Food(Xf.decs, Xfood.dec, Temp, C3[ t - 1])
                else:
                    if np.random.rand() > 0.6:
                        newXf_dec = self.so_fight(Xf.decs,Xf.objs, Xm.bestInd.dec, Xm.bestInd.obj, t1[ t - 1], C3[ t - 1], Q_val)
                    else:
                        _, newXf_dec = self.so_mating(Xm.decs, Xf.decs,Xm.objs, Xf.objs, C3[ t - 1], Q_val, lb, ub)

            # 随机替换最差的解
            index = np.argsort(Xm.objs)
            index1 = np.argsort(Xf.objs)
            for i in range(round(Xf.decs.shape[0] / 10)):
                newXm_dec[index[-(i + 1)], :] = lb + np.random.rand() * (ub - lb)
                newXf_dec[index1[-(i + 1)], :] = lb + np.random.rand() * (ub - lb)

            # 评估解并更新
            Xm,  reward_m, failure_times_m = self.evaluation_reward(Xm, newXm_dec, lb, ub, fobj, failure_times_m)
            Xf,  _, failure_times_f = self.evaluation_reward(Xf, newXf_dec,  lb, ub, fobj, failure_times_f)

            # 获取下一步的状态
            next_state_m = self.get_state_knn(Xm.decs, Xm.objs, round(N/10))

            # 更新 Q 表
            for i in range(Xm.decs.shape[0]):
                q_table_m = self.update_q_table(state_m[i, :], action_m[i], reward_m[i], next_state_m[i, :], q_table_m)
            Xfood = Xm.bestInd if Xm.bestInd.obj < Xf.bestInd.obj else Xf.bestInd

            # 记录全局最优适应度
            gbest_t[ t] = Xfood.obj
            print("Iter:{t} Best:{GYbest}",t,Xfood.obj)
        fval = Xfood.obj
        return Xfood,fval,gbest_t

