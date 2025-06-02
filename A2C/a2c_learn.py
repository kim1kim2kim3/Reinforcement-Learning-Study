import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import gym
import matplotlib.pyplot as plt


# =======================================
#  Actor 네트워크 정의 (Gaussian Policy)
# =======================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        # 은닉층 정의
        self.h1 = nn.Linear(state_dim, 64)
        self.h2 = nn.Linear(64, 32)
        self.h3 = nn.Linear(32, 16)

        # 평균(mu) 출력층 (tanh -> [-1,1], 이후 action_bound 곱)
        self.mu_layer = nn.Linear(16, action_dim)
        # 표준편차(std) 출력층 (softplus -> 양수)
        self.std_layer = nn.Linear(16, action_dim)

    def forward(self, x):
        """
        입력 상태 x: tensor of shape [batch_size, state_dim]
        출력:
            mu: tensor [batch_size, action_dim], in [-action_bound, +action_bound]
            std: tensor [batch_size, action_dim], in (0, ∞)
        """
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))

        mu = torch.tanh(self.mu_layer(x))
        mu = mu * self.action_bound  # [-action_bound, +action_bound]로 스케일링

        std = F.softplus(self.std_layer(x))
        return mu, std


# =======================================
#  Critic 네트워크 정의 (V-value approximator)
# =======================================
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # 은닉층 정의
        self.h1 = nn.Linear(state_dim, 64)
        self.h2 = nn.Linear(64, 32)
        self.h3 = nn.Linear(32, 16)
        # V-value 출력(스칼라)
        self.v_layer = nn.Linear(16, 1)

    def forward(self, x):
        """
        입력 상태 x: tensor of shape [batch_size, state_dim]
        출력: V(x): tensor of shape [batch_size, 1]
        """
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        v = self.v_layer(x)
        return v


# =======================================
#  A2C Agent 정의
# =======================================
class A2CAgent:
    def __init__(self, env):
        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LR = 1e-4
        self.CRITIC_LR = 1e-3

        self.env = env
        # 상태 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동 범위
        self.action_bound = float(env.action_space.high[0])
        # std_clipping 범위
        self.std_bound = [1e-2, 1.0]

        # Actor 및 Critic 네트워크 생성
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim)

        # GPU 사용 가능하면 GPU로 이동
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        # Optimizer
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.CRITIC_LR)

        # 에피소드 리워드 저장용
        self.save_epi_reward = []

    def log_pdf(self, mu, std, action):
        """
        평균 mu, 표준편차 std, 행동 action 에 대한 log-likelihood 계산
        (정규분포 가정). 배치 단위로 계산하며,
        출력 shape: [batch_size, 1]
        """
        # std를 지정한 범위로 클리핑
        std_clipped = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        var = std_clipped.pow(2)

        # 정규분포 로그밀도: -0.5 * ((a - mu)^2 / var + log(2π var))
        log_prob = -0.5 * ((action - mu) ** 2) / var \
                   - 0.5 * torch.log(var * 2 * np.pi)

        # 다차원 행동인 경우 action_dim 차원을 합(sum)으로 축소
        # 결과 shape: [batch_size, 1]
        return log_prob.sum(dim=1, keepdim=True)

    def get_action(self, state_np):
        """
        환경(state_np: numpy [state_dim]) 에서 행동을 샘플링하고 반환
        반환: numpy array shape [action_dim]
        """
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        mu, std = self.actor(state_tensor)
        mu = mu.detach().cpu().numpy().squeeze(0)
        std = std.detach().cpu().numpy().squeeze(0)
        # std 클리핑
        std = np.clip(std, self.std_bound[0], self.std_bound[1])

        # Gaussian 샘플링
        action = np.random.normal(mu, std, size=self.action_dim)
        # 행동 범위 내로 클리핑
        return np.clip(action, -self.action_bound, self.action_bound)

    def critic_learn(self, states, td_targets):
        """
        Critic 업데이트 (MSE Loss)
        states: torch.FloatTensor [batch_size, state_dim]
        td_targets: torch.FloatTensor [batch_size, 1]
        """
        self.critic_opt.zero_grad()
        values = self.critic(states)  # [batch_size,1]
        loss = F.mse_loss(td_targets, values)
        loss.backward()
        self.critic_opt.step()

    def actor_learn(self, states, actions, advantages):
        """
        Actor 업데이트 (Policy Gradient Loss)
        states: torch.FloatTensor [batch_size, state_dim]
        actions: torch.FloatTensor [batch_size, action_dim]
        advantages: torch.FloatTensor [batch_size, 1]
        """
        self.actor_opt.zero_grad()
        mu, std = self.actor(states)  # mu/std shape: [batch_size, action_dim]
        log_policy_pdf = self.log_pdf(mu, std, actions)  # [batch_size,1]

        # loss = - sum(log_prob * advantage)
        # ( 원본: loss_policy = log_pdf * advantages, loss = sum(-loss_policy) )
        loss = -torch.sum(log_policy_pdf * advantages)
        loss.backward()
        self.actor_opt.step()

    def td_target(self, rewards, next_v_values, dones):
        """
        TD 타깃 계산: y_i = r_i + gamma * V(next)
        rewards: numpy [batch_size, 1]
        next_v_values: numpy [batch_size, 1]
        dones: numpy [batch_size, 1] (bool 또는 0/1)
        반환: torch.FloatTensor [batch_size,1]
        """
        batch_size = rewards.shape[0]
        y = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            if dones[i]:
                y[i, 0] = rewards[i, 0]
            else:
                y[i, 0] = rewards[i, 0] + self.GAMMA * next_v_values[i, 0]
        return torch.from_numpy(y).float().to(self.device)

    def unpack_batch(self, batch_list):
        """
        리스트 형태의 배열(각 요소: shape [1, dim])을 하나의 numpy 배열([batch_size, dim])로 합침
        ex) batch_list = [np.array([[s1]]), np.array([[s2]]), ...]
        """
        return np.concatenate(batch_list, axis=0)

    def train(self, max_episode_num):
        """
        A2C 전체 학습 루프
        """
        for ep in range(int(max_episode_num)):
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []

            episode_reward = 0.0
            done = False

            # 환경 초기화
            state = self.env.reset()
            if isinstance(state, tuple):
                # gymnasium 일 경우 (obs, info) 튜플 반환 가능
                state = state[0]

            while not done:
                # 행동 선택
                action = self.get_action(state)  # numpy [action_dim]

                # 환경 한 스텝 진행
                next_step = self.env.step(action)
                if len(next_step) == 5:
                    # 일부 Gym 환경 (gymnasium) 는 (obs, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, info = next_step
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = next_step

                # (state, action, reward, next_state, done)를 배치로 저장
                s_np = np.reshape(state, [1, self.state_dim])               # [1, state_dim]
                a_np = np.reshape(action, [1, self.action_dim])            # [1, action_dim]
                # 보상 정규화 (원본: (reward + 8) / 8)
                r_np = np.reshape((reward + 8) / 8.0, [1, 1])               # [1, 1]
                ns_np = np.reshape(next_state, [1, self.state_dim])        # [1, state_dim]
                d_np = np.reshape(done, [1, 1])                             # [1, 1] (bool->numpy)

                batch_states.append(s_np)
                batch_actions.append(a_np)
                batch_rewards.append(r_np)
                batch_next_states.append(ns_np)
                batch_dones.append(d_np)

                # 에피소드 리워드 축적
                episode_reward += reward

                # 배치가 가득 찼으면 학습 수행
                if len(batch_states) >= self.BATCH_SIZE:
                    # numpy 배열로 변환
                    states_np       = self.unpack_batch(batch_states)        # [B, state_dim]
                    actions_np      = self.unpack_batch(batch_actions)       # [B, action_dim]
                    rewards_np      = self.unpack_batch(batch_rewards)       # [B, 1]
                    next_states_np  = self.unpack_batch(batch_next_states)   # [B, state_dim]
                    dones_np        = self.unpack_batch(batch_dones)         # [B, 1]

                    # 다음 상태에 대한 V값 계산 (torch tensor)
                    next_states_tensor = torch.from_numpy(next_states_np).float().to(self.device)
                    next_v_values = self.critic(next_states_tensor).detach().cpu().numpy()  # [B,1], numpy

                    # TD 타겟 계산 (torch tensor)
                    td_targets = self.td_target(rewards_np, next_v_values, dones_np)  # [B,1] tensor

                    # Critic 업데이트
                    states_tensor = torch.from_numpy(states_np).float().to(self.device)
                    self.critic_learn(states_tensor, td_targets)

                    # Advantage 계산: A = r + γ V(next) - V(curr)
                    v_values = self.critic(states_tensor).detach()                           # [B,1] tensor
                    next_v_tensor = torch.from_numpy(next_v_values).float().to(self.device)  # [B,1] tensor
                    rewards_tensor = torch.from_numpy(rewards_np).float().to(self.device)    # [B,1]
                    dones_tensor = torch.from_numpy(dones_np.astype(np.float32)).to(self.device)  # [B,1]
                    # 다만 dones는 A2C에서 advantage에 그대로 사용하므로 dones 역할은 계산식에 이미 반영됐음
                    advantages = rewards_tensor + self.GAMMA * next_v_tensor * (1 - dones_tensor) - v_values  # [B,1]

                    # Actor 업데이트
                    actions_tensor = torch.from_numpy(actions_np).float().to(self.device)    # [B, action_dim]
                    self.actor_learn(states_tensor, actions_tensor, advantages)

                    # 배치 초기화
                    batch_states = []
                    batch_actions = []
                    batch_rewards = []
                    batch_next_states = []
                    batch_dones = []

                # 상태 업데이트
                state = next_state if not isinstance(next_state, tuple) else next_state[0]

            # 한 에피소드 종료 시 출력
            print(f"Episode: {ep+1}, Reward: {episode_reward:.2f}")
            self.save_epi_reward.append(episode_reward)

            # 10 에피소드마다 모델 저장
            if (ep + 1) % 10 == 0:
                torch.save(self.actor.state_dict(), "./save_weights/pendulum_actor.pth")
                torch.save(self.critic.state_dict(), "./save_weights/pendulum_critic.pth")

        # 에피소드 리워드 기록 파일로 저장
        np.savetxt('./save_weights/pendulum_epi_reward.txt', np.array(self.save_epi_reward))
        print("Training finished. Episode rewards:", self.save_epi_reward)

    def plot_result(self):
        """
        학습 결과 리워드 그래프 출력
        """
        plt.plot(self.save_epi_reward)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("A2C Training Rewards")
        plt.show()


# ====================
#  실제 학습 실행 예시
# ====================
if __name__ == "__main__":
    # 예시: Pendulum-v1 환경
    env = gym.make("Pendulum-v1")
    agent = A2CAgent(env)

    max_episodes = 200  # 원하는 에피소드 수로 조절
    agent.train(max_episodes)
    agent.plot_result()
