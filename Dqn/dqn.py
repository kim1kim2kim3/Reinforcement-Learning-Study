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
SEED                = 42
ENV_NAME            = "CarRacing-v3"   
TOTAL_EPISODES      = 1000            # 총 에피소드 수 
MAX_STEPS_PER_EP    = 1000            # 1000프레임 후 게임 종료
REPLAY_CAPACITY     = 100_000         # 리플레이 버퍼 크기
BATCH_SIZE          = 64              # 배치 크기
GAMMA               = 0.99            # 할인 계수
LEARNING_RATE       = 1e-4            # 학습률
EPS_START           = 1.0             # 초기 ε 값
EPS_END             = 0.05            # 최종 ε 값
EPS_DECAY_FRAMES    = 500_000         # ε 감소 주기
TARGET_SYNC_EVERY   = 5000            # 타겟 네트워크 동기화 주기
SAVE_EVERY_STEPS    = 10_000          # 체크포인트 저장 주기
STACK_LEN           = 4               # 상태 스택 길이
LOG_PATH            = "training_log.csv"
SAVE_DIR            = "models"
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """RGB (96×96×3)을 float32 그레이스케일 (1×84×84)로 변환하고 [0,1] 범위로 정규화"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame[None, ...]  # 채널 차원 추가 → (1,84,84)


# ────────────────────────────────────────────────────────────────────
# 3. Replay buffer
# ────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity) # 이것을 통해 버퍼가 capacity 만큼 저장되고 초과되면 push()호출 시 가장 오래된 데이터가 삭제됨

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = map(np.array, zip(*batch))
        return (
            torch.as_tensor(s, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(a, dtype=torch.int64,  device=DEVICE).unsqueeze(1),
            torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ────────────────────────────────────────────────────────────────────Q‑network────────────────────────────────────────────────────────────────────

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
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ────────────────────────────────────────────────────────────────────Agent────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space                                              # 행동 공간
        self.num_actions  = action_space.n                                            # 행동 공간의 크기

        self.policy_net  = DQN(self.num_actions).to(DEVICE)                           # 폴리시 네트워크 생성
        self.target_net  = DQN(self.num_actions).to(DEVICE)                           # 타겟 네트워크 생성
        self.target_net.load_state_dict(self.policy_net.state_dict())                 # 타겟 네트워크 초기화
        self.target_net.eval()                                                        # 타겟 네트워크 평가 모드 설정

        self.buffer      = ReplayBuffer(REPLAY_CAPACITY)                              # 리플레이 버퍼 생성
        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE) # 옵티마이저 생성

        self.steps_done  = 0                                                          # 탐색 횟수 초기화
        self.epsilon     = EPS_START                                                  # ε 초기화

    # ε‑greedy action selection
    def select_action(self, state: np.ndarray):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        with torch.no_grad():
            state_v = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.policy_net(state_v)
            return int(torch.argmax(q_values, dim=1).item())

    # 선형적으로 ε 감소
    def update_epsilon(self):
        self.steps_done += 1
        self.epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (self.steps_done / EPS_DECAY_FRAMES),
        )

    # 학습 단계
    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0                                                                # 버퍼에 충분한 샘플이 없음

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE) # 버퍼에서 랜덤 샘플 추출

        # Q(s,a)
        q_pred = self.policy_net(states).gather(1, actions)
        
        # max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + GAMMA * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item() # 손실 값 반환

    def sync_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ────────────────────────────────────────────────────────────────────메인 함수────────────────────────────────────────────────────────────────────

def main():
    set_global_seed(SEED)

    env = gym.make(ENV_NAME, continuous=False, render_mode="rgb_array", disable_env_checker=True)   # 환경 생성
    env.action_space.seed(SEED)                                                                     # 행동 공간 시드 설정

    agent = Agent(env.action_space)                                                                 # 에이전트 생성

    # CSV 파일 준비
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Episode", "Return", "Epsilon", "Loss", "BufferSize", "Steps", "Timestamp"])

    global_steps = 0
    for ep in range(1, TOTAL_EPISODES + 1):
        
        # 게임 초기화
        obs, _ = env.reset(seed=SEED + ep)

        # 상태 스택 초기화
        state_stack = deque(maxlen=STACK_LEN)

        # 4개의 초기 프레임
        for _ in range(STACK_LEN):
            state_stack.append(preprocess_frame(obs))

        # 상태 스택 연결
        state = np.concatenate(state_stack, axis=0)  # shape (4,84,84)

        ep_return = 0.0
        ep_loss_sum = 0.0
        ep_loss_cnt = 0


        # 게임 실행
        for step in range(MAX_STEPS_PER_EP):                                    # 1000프레임 후 게임 종료
            action = agent.select_action(state)                                 # 행동 선택
            agent.update_epsilon()                                              # ε 감소

            obs_next, reward, terminated, truncated, _ = env.step(action)       # 행동 실행
            done = terminated or truncated
            state_stack.append(preprocess_frame(obs_next))                      # 상태 스택 업데이트
            next_state = np.concatenate(state_stack, axis=0)                    # 상태 스택 연결

            agent.buffer.push(state, action, reward, next_state, done)          # 리플레이 버퍼에 데이터 추가
            loss = agent.learn()                                                # 학습 단계
            
            if loss:                                                            # 손실 값이 있으면
                ep_loss_sum += loss                                             # 손실 합계 업데이트
                ep_loss_cnt += 1                                                # 손실 횟수 업데이트

            state = next_state                                                  # 상태 업데이트
            ep_return += reward                                                 # 보상 합계 업데이트
            global_steps += 1                                                   # 탐색 횟수 업데이트

            # target network sync
            if global_steps % TARGET_SYNC_EVERY == 0:        
                agent.sync_target_net()                                          

            # periodic checkpoint
            if global_steps % SAVE_EVERY_STEPS == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_steps:06d}.pt")
                torch.save(agent.policy_net.state_dict(), ckpt_path)
                print(f"✅ checkpoint saved: {ckpt_path}")

            if done:
                break

        avg_loss = ep_loss_sum / ep_loss_cnt if ep_loss_cnt else 0.0            # 평균 손실 계산

        # 에피소드 종료 시 체크포인트 저장
        torch.save(agent.policy_net.state_dict(), os.path.join(SAVE_DIR, "dqn_carracing_latest.pt"))

        # CSV 파일에 로그 추가
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                ep_return,
                round(agent.epsilon, 4),
                round(avg_loss, 6),
                len(agent.buffer),
                global_steps,
                datetime.now().isoformat(timespec="seconds"),
            ])

        print(
            f"Ep {ep:4d} | R {ep_return:7.2f} | ε {agent.epsilon:.3f} | loss {avg_loss:.4f} | steps {global_steps} | buffer {len(agent.buffer)}"
        )

    env.close()


if __name__ == "__main__":
    main()
