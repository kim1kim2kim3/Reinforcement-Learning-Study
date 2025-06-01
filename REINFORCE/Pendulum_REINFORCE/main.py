import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Normal
from Pendulum import MLPGaussianPolicy
from Pendulum import REINFORCE          

""" #-------------------------------------------------------------------------


# State 값 확인하는 코드
env = gym.make('Pendulum-v1')  # Pendulum-v1 환경 생성
s, _ = env.reset()  # 환경 초기화 후 초기 상태 s와 추가 정보 _를 반환

print("The dimension of state space: ", env.observation_space.shape[0])  # 상태 공간 차원 수 출력
print("The dimension of action space: ", env.action_space.shape[0])  # 행동 공간 차원 수 출력
print("Initial state: ", s)  # 초기 상태 값 출력

# -------------------------------------------------------------------------

# 어떤 신경망의 구조를 가지고 있는지를 알 수 있는 코드
env = gym.make('Pendulum-v1')  # 다시 Pendulum-v1 환경 생성 (정책 네트워크 초기화를 위해)

state_dim = env.observation_space.shape[0]  # 상태 차원 수를 state_dim에 저장
action_dim = env.action_space.shape[0]  # 행동 차원 수를 action_dim에 저장
hidden_dims = (512,)  # 은닉층 크기를 튜플로 정의 (여기서는 512 뉴런 1개)
activation_fn = F.relu  # 활성화 함수로 ReLU 설정

policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims, activation_fn)  # MLP 기반 가우시안 정책 네트워크 생성
print(policy)  # 네트워크 구조 출력

# -------------------------------------------------------------------------
# 행동 선택 과정 코드
s, _ = env.reset()  # 환경 초기화 및 상태 s 재설정
s = torch.as_tensor(s, dtype=torch.float)  # NumPy 배열 s를 PyTorch 텐서로 변환

mu, std = policy(s)  # 정책 네트워크로부터 평균(mu)과 표준편차(std) 출력
a = torch.normal(mu, std)  # 정규분포(N(mu,std))로부터 샘플링하여 행동 a 생성
a = torch.tanh(a)  # 샘플링된 행동을 tanh로 한정하여 (-1,1) 범위로 변환
a = a.cpu().detach().numpy()  # 텐서를 CPU NumPy 배열로 변환

max_action = env.action_space.high  # 행동 공간의 최대값 벡터
min_action = env.action_space.low  # 행동 공간의 최소값 벡터

a = 0.5 * (max_action - min_action) * (a + 1) + min_action  # tanh(-1,1) 범위를 원래 환경 행동 범위로 조정

print("Selected action: ", a)  # 선택된 행동 출력 """


def seed_all(seed):
    
    random.seed(seed)  # 파이썬 랜덤 시드 고정
    np.random.seed(seed)  # NumPy 랜덤 시드 고정
    torch.manual_seed(seed)  # CPU 텐서 연산용 시드 고정
    torch.cuda.manual_seed(seed)  # 단일 GPU 사용 시 시드 고정
    torch.cuda.manual_seed_all(seed)  # 다중 GPU 사용 시 시드 고정
    
    # cuDNN 최적화 기능 비활성화하여 결정적 결과 보장
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(env_name, agent, seed, eval_iterations):
    
    env = gym.make(env_name)  # 평가용 환경 생성
    scores = []  # 에피소드별 리턴 저장 리스트
    
    for i in range(eval_iterations):  # 지정된 횟수만큼 평가 반복
        
        # 환경 초기화, seed를 에피소드마다 다르게 설정해 평가 안정화
        (s, _), terminated, truncated, score = env.reset(seed=seed + 100 + i), False, False, 0
        
        # 에피소드가 끝날 때까지 행동 반복
        while not (terminated or truncated):
            
            a = agent.act(s, training=False)  # 평가 모드로 행동 선택
            
            # 환경에 행동 전달: 정책에서 반환된 값을 환경 액션 범위에 맞춰서 전달
            s_prime, r, terminated, truncated, _ = env.step(2.0 * a)
            score += r  # 누적 보상 갱신
            s = s_prime  # 다음 상태로 전환
            
        scores.append(score)  # 한 에피소드의 리턴 저장
        
    env.close()  # 환경 종료
    
    return round(np.mean(scores), 4)  # 평균 리턴을 소수점 4자리로 반올림하여 반환


# 환경 이름 지정
env_name = 'Pendulum-v1'


# 학습 하이퍼파라미터 및 시드 설정

seed = 1  # 재현성을 위한 시드
seed_all(seed)  # 모든 라이브러리 시드 일괄 고정
max_iterations = 1000000  # 전체 학습 스텝 수
eval_intervals = 10000  # 평가 수행 간격 (스텝 단위)
eval_iterations = 10  # 평가 시 에피소드 수
gamma = 0.9  # 할인율
lr = 0.0003  # 학습률

# 환경 및 에이전트 초기화
env = gym.make(env_name)  # 학습 환경 생성
state_dim = env.observation_space.shape[0]  # 상태 차원 수
action_dim = env.action_space.shape[0]  # 행동 차원 수
agent = REINFORCE(state_dim, action_dim, gamma, lr)  # REINFORCE 에이전트 생성

# 학습 로그 저장용 리스트 초기화
logger = []

# 환경 초기 상태 및 종료 플래그 초기화
(s, _), terminated, truncated = env.reset(seed=seed), False, False

# 학습 루프 시작
for t in range(1, max_iterations + 1):
    
    a = agent.act(s)  # 현재 정책에 따라 행동 선택
    
    # 행동을 환경의 실제 액션 범위(예: [-2,2])로 변환하여 step 수행
    s_prime, r, terminated, truncated, _ = env.step(2.0 * a)
    
    # 선택한 transition을 에이전트에 전달하여 학습 단위 처리
    result = agent.process((s, a, r, s_prime, terminated, truncated))
    
    s = s_prime  # 다음 상태로 전환
    
    if result is not None:
        # 정책 업데이트 후 반환된 손실 값을 로거에 기록
        logger.append([t, 'policy_loss', result['policy_loss']])
    
    if terminated or truncated:
        # 에피소드 종료 시 환경 재설정 및 종료 플래그 초기화
        (s, _), terminated, truncated = env.reset(), False, False
        
    if t % eval_intervals == 0:
        # 일정 간격마다 평가 수행 및 평균 리턴 기록
        score = evaluate(env_name, agent, seed, eval_iterations)
        logger.append([t, 'Avg return', score])

logger = pd.DataFrame(logger)
logger.columns = ['step', 'key', 'value']

fig = plt.figure(figsize=(12, 4))

ax = fig.add_subplot(1, 2, 1)
key = 'Avg return'
ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'bo-')
ax.grid(axis='y')
ax.set_title("Average return over 10 episodes")
ax.set_xlabel("Step")
ax.set_ylabel("Avg return")

ax = fig.add_subplot(1, 2, 2)
key = 'policy_loss'
ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
ax.grid(axis='y')
ax.set_title("Policy loss")
ax.set_xlabel("Step")
ax.set_ylabel("Policy loss")

fig.tight_layout()
plt.show()