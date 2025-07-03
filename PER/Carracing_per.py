import os
import csv
import random
from collections import deque
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ────────────────────────────────────────────────────────────────────
# 1. Global configuration
# ────────────────────────────────────────────────────────────────────
SEED = 42
ENV_NAME = "CarRacing-v3"  # discrete: continuous=False by default
TOTAL_EPISODES = 1000  # adjust as needed
MAX_STEPS_PER_EP = 1000  # CarRacing ends after ~1000 frames by default
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRAMES = 500_000  # linear decay over this many env steps
TARGET_SYNC_EVERY = 5000  # sync target net parameters
SAVE_EVERY_STEPS = 10_000  # checkpoint interval
STACK_LEN = 4
MIN_BUFFER_SIZE = 10_000  # PER 학습 시작을 위한 최소 버퍼 크기

# PER hyper‑params
ALPHA = 0.6  # how much prioritization is used (0 = uniform)
BETA_START = 0.4  # importance‑sampling weight at start
BETA_FRAMES = 1_000_000  # frames over which beta → 1.0

LOG_PATH = "training_log_per.csv"
SAVE_DIR = "models_per" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# 2. Utility functions
# ────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# grayScale + 사이즈 축소 + 정규화
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame[None, ...]  # 채널 차원 추가 84x84 -> 1x84x84


#---------------------------------------PER 구현 부분-------------------------------------------------
class PrioritizedReplayBuffer:
    
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity                                            # 버퍼 최대 크기(초과시 덮어씀)
        self.alpha = alpha                                                  # 파라미터 α
        self.buffer = []                                                    # 튜플 저장 리스트
        self.priorities = np.zeros((capacity,), dtype=np.float32)           # 각 전이의 우선순위 저장 배열(td값 저장 배열이라 생각하면 됨)
        self.pos = 0                                                        # 다음에 저장할 위치 나타냄


    def push(self, state, action, reward, next_state, done):
        
        max_prio = self.priorities.max() if self.buffer else 1.0                # 현재 버퍼가 비어있으면 priority 기본값을 1.0, 아니면 현재 버퍼에서 가장 큰 priority 사용
        
        if len(self.buffer) < self.capacity:                                    # 버퍼가 아직 capacity에 도달하지 않은 경우에는 append로 새 경험을 추가
            self.buffer.append((state, action, reward, next_state, done))  
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)   # 버퍼 다 차면, 오래된 경험을 새 경험으로 덮어씀(pos 위치에)
            
        self.priorities[self.pos] = max_prio                                    # 새로 저장된 transition의 priority 값을 max_prio로 갱신
        self.pos = (self.pos + 1) % self.capacity                               # pos(저장 위치)를 한 칸 앞으로 원형 큐처럼 이동. capacity를 넘으면 다시 0으로


    def sample(self, batch_size: int, beta: float): 
        
        if len(self.buffer) == self.capacity:                                 # 버퍼가 버퍼 크기만큼 다 찼으면
            prios = self.priorities                                           # priorites 배열 전체 사용
        else:                                               
            prios = self.priorities[:len(self.buffer)]                        # 아니면 priority 배열의 0~버퍼길이-1까지 사용

        probs = prios ** self.alpha                                           # 샘플링 확률에 알파만큼 거듭제곱
        probs /= probs.sum()                                                  # 확률분포로 만들기
 
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)     # 샘플링 확률에 따라 버퍼에서 batch size만큼 인덱스 뽑음
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)                                         # per의 수식 중 weights 계산
        weights /= weights.max()                                                              # 1로 정규화하기(학습의 안정성 위해, IS 엄밀성 깨짐)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)   # 뒤에서 곱해주기 편하도록 파이토치 텐서로 변환

        s, a, r, s_next, d = map(np.array, zip(*samples))                                     # 각 위치별로 값을 묶어서 그룹을 만듬(세로축으로 그룹짓기) and numpy array로 변환
        return (
            torch.as_tensor(s, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1),
            torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            weights,
            indices,
        )
        

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):           # 두 배열의 같은 위치에 있는 값끼리 쌍을 만들어서 풀어서 넣어줌
            self.priorities[idx] = prio                      # 각 전이들을 새로 게산된 TD_Error의 위치를 가르키게 해줌


    def __len__(self):
        return len(self.buffer)            # 버퍼 크기 알려줌


# --------------------------------------------------Q‑network---------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # cnn을 한줄로 펼침(x.size(0)=배치 크기, -1은 자동으로 나머지 차원 다 곱해서 한줄로 펼침)
        return self.fc(x)

# ----------------------------------------------------- Agent--------------------------------------------------------------

class Agent:
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.num_actions = action_space.n

        self.policy_net = DQN(self.num_actions).to(DEVICE)
        self.target_net = DQN(self.num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # policy 네트워크의 현재 파라미터 전체를 가져와, 그 파라미터를 target 네트워크에 복사해 똑같이 맞춤(state_dict = 모델의 모든 파라미터를 담고 있는 딕셔너리)
        self.target_net.eval()                                        # 타겟 네트워크에는 드롭아웃이나 배치 정규화도 학습이 되면 안됨 => eval 붙임

        self.buffer = PrioritizedReplayBuffer(REPLAY_CAPACITY, ALPHA)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE) # polcy_net의 모든 파라미터만 가져와서 이 optimizer가 policy_net의 파라미터만 업데이트하게 됨

        self.steps_done = 0
        self.epsilon = EPS_START


    def select_action(self, state: np.ndarray):
        if random.random() < self.epsilon:                                                             # 입실론을 통한 랜덤 샘플링
            return self.action_space.sample()
        with torch.no_grad():                                                                          # max Q (역전파 안되게 막아야함)
            state_v = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.policy_net(state_v)
            return int(torch.argmax(q_values, dim=1).item())


    def update_epsilon(self):                                                                          # ε 선형 감소
        self.steps_done += 1
        self.epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (self.steps_done / EPS_DECAY_FRAMES),
        )


    def beta_by_frame(self):                                                                           # beta 선형 증가
        return min(1.0, BETA_START + (1.0 - BETA_START) * (self.steps_done / BETA_FRAMES))


    def learn(self):                                                                                   # 최소 버퍼 사이즈보다 작으면 학습 x
        if len(self.buffer) < MIN_BUFFER_SIZE: 
            return 0.0  

        beta = self.beta_by_frame()
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(BATCH_SIZE, beta)

        # Q(s,a)
        q_pred = self.policy_net(states).gather(1, actions)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + GAMMA * q_next * (1 - dones)

        td_errors = q_target - q_pred
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # priority 업데이트 (|δ| + ε)
        new_prios = td_errors.abs().detach().cpu().numpy().flatten() + 1e-6
        self.buffer.update_priorities(indices, new_prios)

        return loss.item()
    
    # target network 동기화시키는 함수
    def sync_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        

#--------------------------------------Main loop-----------------------------------------------------


def main():
    set_global_seed(SEED)
    env = gym.make(ENV_NAME, continuous=False, render_mode="rgb_array", disable_env_checker=True)
    env.action_space.seed(SEED)

    agent = Agent(env.action_space)


    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Episode", "Return", "Epsilon", "Loss", "BufferSize", "Steps", "Timestamp"])

    global_steps = 0
    for ep in range(1, TOTAL_EPISODES + 1):
        obs, _ = env.reset(seed=SEED + ep)
        state_stack = deque(maxlen=STACK_LEN)
        
        for _ in range(STACK_LEN):
            state_stack.append(preprocess_frame(obs))
            
        state = np.concatenate(state_stack, axis=0)

        ep_return = 0.0
        ep_loss_sum = 0.0
        ep_loss_cnt = 0

        for step in range(MAX_STEPS_PER_EP):
            action = agent.select_action(state)
            agent.update_epsilon()

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_stack.append(preprocess_frame(obs_next))
            next_state = np.concatenate(state_stack, axis=0)

            agent.buffer.push(state, action, reward, next_state, done)
            
            # MIN_BUFFER_SIZE 이상 버퍼에 데이터 있을때만 학습
            if len(agent.buffer) >= MIN_BUFFER_SIZE:
                loss = agent.learn()
                if loss:
                    ep_loss_sum += loss
                    ep_loss_cnt += 1

            state = next_state
            ep_return += reward
            global_steps += 1
            
            # 일정 동기화 주기 되면 target network 동기화
            if global_steps % TARGET_SYNC_EVERY == 0:
                agent.sync_target_net()
                
            # 일정 주기되면 checkpoint 모델 생성
            if global_steps % SAVE_EVERY_STEPS == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_steps:06d}.pt")
                torch.save(agent.policy_net.state_dict(), ckpt_path)
                print(f"checkpoint saved: {ckpt_path}")

            if done:
                break

        avg_loss = ep_loss_sum / ep_loss_cnt if ep_loss_cnt else 0.0

        torch.save(agent.policy_net.state_dict(), os.path.join(SAVE_DIR, "dqn_per_carracing_latest.pt"))

        # CSV 로그 기록
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, ep_return, round(agent.epsilon, 4), round(avg_loss, 6), len(agent.buffer), global_steps, datetime.now().isoformat(timespec="seconds")])

        print(f"Ep {ep:4d} | R {ep_return:7.2f} | ε {agent.epsilon:.3f} | loss {avg_loss:.4f} | steps {global_steps} | buffer {len(agent.buffer)}")

        # 첫 번째 에피소드에서 설정 정보 출력
        if ep == 1:
            print("PER 학습을 시작합니다")

    env.close()

if __name__ == "__main__":
    main()
