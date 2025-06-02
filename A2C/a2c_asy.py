import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import gym
from gym.vector import SyncVectorEnv
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
        # 표준편차(std) 출력층 (softplus -> (0, ∞))
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

        mu = torch.tanh(self.mu_layer(x))           # [-1,1]
        mu = mu * self.action_bound                 # [-action_bound, +action_bound]

        std = F.softplus(self.std_layer(x))         # (0, ∞)
        return mu, std


# =======================================
#  Critic 네트워크 정의 (V-value Approximator)
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
#  멀티 에이전트 A2C 구현 클래스
# =======================================
class A2CAgentMulti:
    def __init__(self, env_name: str, num_envs: int = 8, gamma: float = 0.95, actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3, std_bound=(1e-2, 1.0)):
        """
        env_name: Gym 환경 이름 (예: "Pendulum-v1")
        num_envs: 동시 실행할 병렬 환경 수
        gamma: 할인율
        actor_lr: Actor 학습률
        critic_lr: Critic 학습률
        std_bound: std 클리핑 범위 (min, max)
        """
        self.gamma = gamma
        self.num_envs = num_envs
        self.std_bound = std_bound

        # 1) 병렬 환경 생성
        def make_env():
            return gym.make(env_name)

        self.envs = SyncVectorEnv([make_env for _ in range(num_envs)])
        obs_space = self.envs.single_observation_space.shape[0]
        act_space = self.envs.single_action_space.shape[0]
        act_bound = float(self.envs.single_action_space.high[0])

        # 2) 네트워크 초기화
        self.actor = Actor(obs_space, act_space, act_bound)
        self.critic = Critic(obs_space)

        # 3) 장치 설정 (GPU 사용 가능 시 GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        # 4) Optimizer 정의
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 5) 리워드 기록용
        self.batch_avg_rewards = []

    def log_pdf(self, mu: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        평균 mu, 표준편차 std, 행동 action 에 대한 log-likelihood를 계산합니다.
        (정규분포 가정) multi-dim 행동일 때 차원별 로그확률을 합산하여 반환합니다.
        출력 shape: [batch_size, 1]
        """
        std_clipped = torch.clamp(std, self.std_bound[0], self.std_bound[1])  # [batch, action_dim]
        var = std_clipped.pow(2)                                             # [batch, action_dim]

        # 정규분포 로그밀도
        log_prob = -0.5 * ((action - mu) ** 2) / var - 0.5 * torch.log(var * 2 * np.pi)
        return log_prob.sum(dim=1, keepdim=True)  # [batch, 1]

    def train(self, max_updates: int = 1000, log_interval: int = 100):
        """
        A2C 학습 루프 (동기화된 병렬 업데이트)
        max_updates: 전체 업데이트(gradient step) 횟수
        log_interval: 로그 출력 주기 (업데이트 횟수 기준)
        """
        # 병렬 환경 초기화
        obs = self.envs.reset()  # shape: [num_envs, state_dim]

        for update in range(1, max_updates + 1):
            # 1) 현재 배치(obs)로부터 액션 분포 산출
            obs_tensor = torch.from_numpy(obs).float().to(self.device)      # [num_envs, state_dim]
            with torch.no_grad():
                mu, std = self.actor(obs_tensor)                            # [num_envs, action_dim]
                mu_np = mu.cpu().numpy()
                std_np = std.cpu().numpy()
                std_np = np.clip(std_np, self.std_bound[0], self.std_bound[1])

            # 2) Gaussian 샘플링으로 액션을 생성
            actions = np.random.normal(mu_np, std_np)                         # [num_envs, action_dim]
            actions = np.clip(actions, -self.actor.action_bound, self.actor.action_bound)

            # 3) 병렬 환경에서 one-step 실행
            next_obs, rewards, dones, infos = self.envs.step(actions)
            # next_obs: [num_envs, state_dim], rewards: [num_envs], dones: [num_envs]

            # 4) Critic으로부터 다음 상태 가치 V(s')
            next_obs_tensor = torch.from_numpy(next_obs).float().to(self.device)
            with torch.no_grad():
                next_v = self.critic(next_obs_tensor).cpu().numpy()          # [num_envs, 1]

            # 5) TD 타깃 및 Advantage 계산
            rewards = rewards.reshape(-1, 1).astype(np.float32)               # [num_envs, 1]
            dones = dones.astype(np.float32).reshape(-1, 1)                   # [num_envs, 1]

            td_target = np.zeros_like(rewards, dtype=np.float32)             # [num_envs, 1]
            for i in range(self.num_envs):
                if dones[i]:
                    td_target[i, 0] = rewards[i, 0]
                else:
                    td_target[i, 0] = rewards[i, 0] + self.gamma * next_v[i, 0]
            td_target_tensor = torch.from_numpy(td_target).float().to(self.device)  # [num_envs,1]

            # 현재 상태에 대한 V(s) 계산
            v_curr = self.critic(obs_tensor)                                   # [num_envs, 1]
            advantages = td_target_tensor - v_curr                             # [num_envs, 1]

            # 6) Critic 업데이트 (MSE Loss)
            self.critic_opt.zero_grad()
            loss_c = F.mse_loss(v_curr, td_target_tensor)
            loss_c.backward()
            self.critic_opt.step()

            # 7) Actor 업데이트 (Policy Gradient Loss)
            actions_tensor = torch.from_numpy(actions).float().to(self.device)  # [num_envs, action_dim]
            self.actor_opt.zero_grad()
            mu2, std2 = self.actor(obs_tensor)                                  # [num_envs, action_dim]
            logp = self.log_pdf(mu2, std2, actions_tensor)                      # [num_envs, 1]
            loss_a = -torch.sum(logp * advantages.detach())                     # detach()로 Advantage 고정
            loss_a.backward()
            self.actor_opt.step()

            # 8) 다음 상태로 갱신 (done=True인 env는 내부에서 자동 reset)
            obs = next_obs

            # 9) 로그 저장 및 출력
            batch_avg_reward = rewards.mean()
            self.batch_avg_rewards.append(batch_avg_reward)
            if update % log_interval == 0:
                avg_r = float(np.mean(self.batch_avg_rewards[-log_interval:]))
                avg_adv = float(advantages.mean().cpu().item())
                print(f"Update {update}/{max_updates}  "
                      f"BatchAvgReward = {avg_r:.3f}  AvgAdvantage = {avg_adv:.3f}")

        print("Training completed.")

    def plot_rewards(self):
        """
        업데이트 단위로 기록된 평균 리워드를 그립니다.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.batch_avg_rewards, linewidth=1)
        plt.xlabel("Update Step")
        plt.ylabel("Batch Average Reward")
        plt.title("A2C Multi-Env Training Progress")
        plt.grid(True)
        plt.show()


# ================
#  스크립트 실행부
# ================
if __name__ == "__main__":
    # 설정
    ENV_NAME = "Pendulum-v1"   # 사용할 Gym 환경
    NUM_ENVS = 8               # 병렬 환경 수
    MAX_UPDATES = 2000         # 총 업데이트 횟수
    LOG_INTERVAL = 200         # 몇 업데이트마다 로그 출력할지

    # 에이전트 생성
    agent = A2CAgentMulti(env_name=ENV_NAME, num_envs=NUM_ENVS,
                          gamma=0.95, actor_lr=1e-4, critic_lr=1e-3,
                          std_bound=(1e-2, 1.0))

    # 학습 시작
    agent.train(max_updates=MAX_UPDATES, log_interval=LOG_INTERVAL)

    # 학습 결과 시각화
    agent.plot_rewards()
