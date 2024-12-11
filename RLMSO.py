import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
class RLMSO:
    def __init__(self, Threshold=0.25,Thresold2= 0.6,C1=0.5,C2=0.05,C3=2):
        self.Threshold=Threshold
        self.Thresold2=Thresold2
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.Nm =None
        self.Nf =None
    def divide_swarm(self, X, fitness):
        self.Nm = round(len(X) / 2)  # eq.(2&3)
        self.Nf = len(X) - self.Nm
        Xm = X[:self.Nm, :]
        Xf = X[self.Nm:, :]
        fitness_m = fitness[:self.Nm]
        fitness_f = fitness[self.Nm:]
        fitnessBest_m = np.min(fitness_m)
        gbest1 = np.argmin(fitness_m)  # Index of the best male
        Xbest_m = Xm[gbest1, :]

        fitnessBest_f = np.min(fitness_f)
        gbest2 = np.argmin(fitness_f)  # Index of the best female
        Xbest_f = Xf[gbest2, :]
        GYbest = np.min(fitness)
        gbest = np.argmin(fitness)  # Index of the global best
        Xfood = X[gbest, :]
        return Xm, Xf, fitness_m, fitness_f,Xbest_m,Xbest_f,fitnessBest_m,fitnessBest_f,Xfood,GYbest
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
    
    def ind_exploration_NoFood(self,X_dec, fitness, ind_fitness, C2, lb, ub):
        vec_flag = np.array([1, -1])  # Equivalent to MATLAB's [1, -1]
        N, dim = X_dec.shape
        newX_dec = np.zeros(dim)  # 1D array for new decision variables

        for j in range(dim):
            rand_leader_index = int(N * np.random.rand())  # Random index in range [0, N-1]
            X_rand = X_dec[rand_leader_index, :]
            flag_index = int(2 * np.random.rand())  # Random index in range [0, 1]
            Flag = vec_flag[flag_index]
            A = np.exp(-fitness[rand_leader_index] / (ind_fitness + np.finfo(float).eps))  # Avoid division by zero
            newX_dec[j] = X_rand[j] + Flag * C2 * A * ((ub[j] - lb[j]) * np.random.rand() + lb[j])
    
        return newX_dec
    import numpy as np

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

    def evaluation_reward(self,X, newX_dec, fitness, lb, ub, fobj, failure_times):
        N = X.shape[0]  # 获取个体数量
        fitness_new = np.zeros(fitness.shape)  # 新的适应度
        fitness_old = fitness.copy()  # 旧的适应度
        reward = -1 * np.ones(N)  # 初始奖励为 -1
        newX_dec = np.clip(newX_dec, lb, ub)
        
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
        for j in range(N):
            # 计算新的适应度
            # fitness_new[j] = fobj(newX_dec[j, :])
            
            # 选择优解
            if fitness_new[j] < fitness[j]:
                X[j, :] = newX_dec[j, :]
                fitness[j] = fitness_new[j]
                reward[j] = 1  # 适应度提高，奖励为 1
                failure_times[j] = 0  # 重置失败次数
            else:
                failure_times[j] += 1  # 适应度未提高，失败次数加 1

        return X, fitness, reward, failure_times
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

    def optimize(self,N,T,lb,ub,dim,fobj):
        Threshold = self.Threshold
        Threshold2 = self.Thresold2
        C1 = self.C1 * np.ones(T)
        C2 = self.C2 * np.ones(T)
        C3 = self.C3 * np.ones(T)
        t1 = np.ones(T)
        ## Init
        X,lb,ub= self.init_pop(N,lb,ub,dim)
        fitness = np.zeros(N)
        for i in range(N):
            fitness[i] = fobj(X[i, :])
        gbest_t = np.zeros(T)
        
        Xm, Xf, fitness_m, fitness_f,Xbest_m,Xbest_f,fitnessBest_m,fitnessBest_f,Xfood,GYbest = self.divide_swarm(X, fitness)
        q_table_m = self.init_Q_parameter()
        

        # Equivalent to MATLAB's empty arrays
        X_out = []
        fitness_out = []

        # Equivalent to MATLAB's zeros
        failure_times_m = np.zeros((self.Nm, 1))  # Assuming Nm is defined
        failure_times_f = np.zeros((self.Nf, 1))  # Assuming Nf is defined

        maxFailure_times = 10

        for t in range(1, T ):
            Temp = np.exp(-t / T)  # eq.(4)
            Q_val = C1[t - 1] * np.exp((t - T) / T)  # eq.(5)

            # 找到超过最大失败次数的个体
            indices = np.where(failure_times_m >= maxFailure_times)[0]
            indices1 = np.where(failure_times_f >= maxFailure_times)[0]

            if len(indices) > 0 or len(indices1) > 0:
                if len(indices) > 0:
                    X_out = np.vstack([X_out, Xm[indices, :]]) if len(X_out) > 0 else Xm[indices, :]
                    fitness_out = np.hstack([fitness_out, fitness_m[indices]]) if len(fitness_out) > 0 else fitness_m[indices]

                if len(indices1) > 0:
                    X_out = np.vstack([X_out, Xf[indices1, :]]) if len(X_out) > 0 else Xf[indices1, :]
                    fitness_out = np.hstack([fitness_out, fitness_f[indices1]]) if len(fitness_out) > 0 else fitness_f[indices1]

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

                xall = np.vstack([X_out, Xm, Xf])
                fitnesspbest = np.hstack([fitness_out, fitness_m, fitness_f])

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
                    fitness_m[indices[i]] = y
                    Xm[indices[i], :] = tempXm

                for i in range(len(indices1)):
                    r = np.random.choice(xall.shape[0], 2, replace=False)
                    x_selected = xall[r, :]
                    randp = np.random.randint(0, p)
                    tempXf = xpbest[randp, :] + F * (x_selected[0, :] - x_selected[1, :])

                    # 边界检查
                    tempXf = np.clip(tempXf, lb, ub)
                    y = fobj(tempXf)
                    fitness_f[indices1[i]] = y
                    Xf[indices1[i], :] = tempXf

            # 获取 Q-learning 状态和行动
            state_m = self.get_state_knn(Xm, fitness_m, round(N/10))
            action_m = self.get_action(q_table_m, state_m)
            newXm_dec = np.zeros_like(Xm)

            # 排序并更新
            index = np.argsort(fitness_m)
            index1 = np.argsort(fitness_f)

            for i in range(Xm.shape[0]):
                if action_m[i] == 1:
                    newXm_dec[i, :] = self.ind_exploration_NoFood(Xm, fitness_m, fitness_m[i], C2[t - 1], lb, ub)
                elif action_m[i] == 2:
                    newXm_dec[i, :] = self.exploit_Food(Xm[i, :].reshape(1, -1), Xfood, Temp, C3[t - 1])
                elif action_m[i] == 3:
                    newXm_dec[i, :] = self.so_fight(Xm[i, :].reshape(1, -1), fitness_m[i].reshape(1, ), Xbest_f, fitnessBest_f, t1[t - 1], C3[t - 1], Q_val)
                else:
                    newXm_dec[i, :], _ = self.so_mating(Xm[i, :].reshape(1, -1), Xf[i, :].reshape(1, -1), fitness_m[i].reshape(1, ), fitness_f[i].reshape(1, ), C3[t - 1], Q_val, lb, ub)

            # 更新 Xf 解
            if Q_val < Threshold:
                newXf_dec = self.exploration_NoFood(Xf, fitness_f, C2[t - 1], lb, ub)
            else:
                if Temp > Threshold2:
                    newXf_dec = self.exploit_Food(Xf, Xfood, Temp, C3[ t - 1])
                else:
                    if np.random.rand() > 0.6:
                        newXf_dec = self.so_fight(Xf, fitness_f, Xbest_m, fitnessBest_m, t1[ t - 1], C3[ t - 1], Q_val)
                    else:
                        _, newXf_dec = self.so_mating(Xm, Xf, fitness_m, fitness_f, C3[ t - 1], Q_val, lb, ub)

            # 随机替换最差的解
            for i in range(round(Xf.shape[0] / 10)):
                newXm_dec[index[-(i + 1)], :] = lb + np.random.rand() * (ub - lb)
                newXf_dec[index1[-(i + 1)], :] = lb + np.random.rand() * (ub - lb)

            # 评估解并更新
            Xm, fitness_m, reward_m, failure_times_m = self.evaluation_reward(Xm, newXm_dec, fitness_m, lb, ub, fobj, failure_times_m)
            Xf, fitness_f, _, failure_times_f = self.evaluation_reward(Xf, newXf_dec, fitness_f, lb, ub, fobj, failure_times_f)

            # 获取下一步的状态
            next_state_m = self.get_state_knn(Xm, fitness_m, round(N/10))

            # 更新 Q 表
            for i in range(Xm.shape[0]):
                q_table_m = self.update_q_table(state_m[i, :], action_m[i], reward_m[i], next_state_m[i, :], q_table_m)

            # 更新全局最优解
            Xbest_m, Xbest_f, fitnessBest_m, fitnessBest_f, GYbest, Xfood = self.updateXbest(Xm, Xf, fitness_m, fitness_f, Xbest_m, Xbest_f, fitnessBest_m, fitnessBest_f)

            # 记录全局最优适应度
            gbest_t[ t] = GYbest
            print("Iter:{t} Best:{GYbest}")
        fval = GYbest
        return Xfood,fval,gbest_t


