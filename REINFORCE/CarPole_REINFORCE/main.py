from CartPole import REINFORCE                     # REINFORCE 알고리즘 에이전트 클래스
import random                                     # 난수 시드 고정용
import numpy as np                                # 수치 계산용
import pandas as pd                               # 로그 데이터 프레임 처리용
import torch                                      # PyTorch 메인 모듈
import torch.nn as nn                             # 신경망 구성용
import torch.nn.functional as F                   # 활성화 함수 등
import gymnasium as gym                           # 강화학습 환경
import matplotlib.pyplot as plt                   # 학습 결과 시각화
from torch.distributions import Normal            # 연속 확률 분포용 (사용되지 않는 경우도 있음)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 자동 최적화 비활성화 → 결정적 결과 보장
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(env_name, agent, seed, eval_iterations):

    env = gym.make(env_name)   # 평가 전용 환경 생성
    scores = []
    
    for i in range(eval_iterations):
        
        # env.reset: (state, info) 반환 → 상태만 받기
        (s, _), terminated, truncated, score = env.reset(seed=seed + 100 + i), False, False, 0
        
        # 한 에피소드 진행
        while not (terminated or truncated):
            a = agent.act(s, training=False)        # 결정론적(policy only) 행동 선택
            s_prime, r, terminated, truncated, _ = env.step(a[0])  # 행동 수행
            score += r                              # 에피소드 보상 누적
            s = s_prime                             # 상태 업데이트
            
        scores.append(score)
        
    env.close()
    
    return round(np.mean(scores), 4)  # 평균 return 반올림하여 반환


# --------------------- 메인 파트 --------------------- #
env_name = 'CartPole-v1'                       # 사용할 환경 이름
seed = 1                                       # 전역 시드
seed_all(seed)                                 # 시드 일괄 고정

# 학습 설정
max_iterations = 100000     # 총 시간 스텝 수
eval_intervals = 5000       # 몇 스텝마다 평가할지
eval_iterations = 10        # 평가 시 에피소드 수
gamma = 0.99                # 할인율
lr = 0.001                  # 학습률

# 환경 및 에이전트 초기화
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]  # 상태 차원 (4)
action_dim = env.action_space.n             # 행동 차원 (2)
agent = REINFORCE(state_dim, action_dim, gamma, lr)  # REINFORCE 에이전트 생성

# 로그 저장용 리스트
logger = []

# 초기 에피소드 상태 리셋
(s, _), terminated, truncated = env.reset(seed=seed), False, False

# --------------------- 학습 루프 --------------------- #
for t in range(1, max_iterations + 1):
    
    a = agent.act(s)  # 정책에 따라 행동 샘플링
    
    # 행동 수행 → 다음 상태, 보상, 종료 정보 획득
    s_prime, r, terminated, truncated, _ = env.step(a[0])
    
    # 경험 버퍼에 경험 추가
    result = agent.process((s, a, r, s_prime, terminated, truncated))
    
    s = s_prime  # 상태 업데이트

    # policy_loss가 반환되면 로그에 기록
    if result is not None:
        logger.append([t, 'policy_loss', result['policy_loss']])

    # 에피소드 종료 시 새로 reset
    if terminated or truncated:
        (s, _), terminated, truncated = env.reset(), False, False

    # eval_intervals마다 성능 평가하고 기록
    if t % eval_intervals == 0:
        score = evaluate(env_name, agent, seed, eval_iterations)
        logger.append([t, 'Avg return', score])

# --------------------- 로그 데이터 프레임화 --------------------- #
logger = pd.DataFrame(logger, columns=['step', 'key', 'value'])

# --------------------- 시각화 --------------------- #
fig = plt.figure(figsize=(12, 4))

# 왼쪽: 평균 return 변화 추이
ax = fig.add_subplot(1, 2, 1)
key = 'Avg return'
ax.plot(
    logger.loc[logger['key'] == key, 'step'],
    logger.loc[logger['key'] == key, 'value'],
    'bo-'
)
ax.grid(axis='y')
ax.set_title("Average return over 10 episodes")
ax.set_xlabel("Step")
ax.set_ylabel("Avg return")

# 오른쪽: policy loss 변화 추이
ax = fig.add_subplot(1, 2, 2)
key = 'policy_loss'
ax.plot(
    logger.loc[logger['key'] == key, 'step'],
    logger.loc[logger['key'] == key, 'value'],
    'b-'
)
ax.grid(axis='y')
ax.set_title("Policy loss")
ax.set_xlabel("Step")
ax.set_ylabel("Policy loss")

fig.tight_layout()
plt.show()
