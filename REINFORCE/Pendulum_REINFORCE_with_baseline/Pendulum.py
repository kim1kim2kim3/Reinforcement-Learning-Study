import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Normal


# 상태 가치 함수 네트워크
class MLPStateValue(nn.Module):

    def __init__(self, state_dim,
                 hidden_dims=(512, ),
                 activation_fn=F.relu):
        
        super(MLPStateValue, self).__init__()

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):

            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.activation_fn = activation_fn
        
    def forward(self, s):

        s = self.activation_fn(self.input_layer(s))

        for hidden_layer in self.hidden_layers:
            s = self.activation_fn(hidden_layer(s))

        s = self.output_layer(s)

        
        return s
    


# 연속 행동 공간을 가진 정책 네트워크
class MLPGaussianPolicy(nn.Module):
    
    def __init__(self, 
                 dim_state, 
                 dim_action, 
                 dim_hiddens=(512,), 
                 activation_fn=F.relu):
        
        # 부모 클래스의 생성자 호출
        super(MLPGaussianPolicy, self).__init__()
        
        # 입력층: 상태 차원(state_dim) → 첫 번째 은닉층 크기(hidden_dims[0])
        self.input_layer = nn.Linear(dim_state, dim_hiddens[0])
        
        # 추가 은닉층들을 ModuleList로 관리(명시적으로 신경망 선언하는게 아니라 for문처럼 동적으로 만들어버리고 싶을때 사용)
        self.hidden_layers = nn.ModuleList()
        
        # for문으로 은닉층 추가
        for i in range(len(dim_hiddens) - 1):
            
            # hidden_dims[i] → hidden_dims[i+1] 형태의 선형 레이어 추가 ex) input→512, 512→action_dim, 만약 (512,514)였다면? input->512->514->action_dim
            hidden_layer = nn.Linear(dim_hiddens[i], dim_hiddens[i+1])
            
            # 은닉층 추가
            self.hidden_layers.append(hidden_layer)
            
        # 평균 출력층: 마지막 은닉층 크기(hidden_dims[-1]) → 행동 수(action_dim) 즉 action마다 평균 출력층이 있음
        self.mu_layer = nn.Linear(dim_hiddens[-1], dim_action)
        
        # 표준편차 출력층: 마지막 은닉층 크기(hidden_dims[-1]) → 행동 수(action_dim) 즉 action마다 표준편차 출력층이 있음
        self.log_std_layer = nn.Linear(dim_hiddens[-1], dim_action)
        
        # 활성화 함수 (예: ReLU)
        self.activation_fn = activation_fn

    def forward(self, s):
        
        # 1) 입력층 처리 및 활성화
        x = self.activation_fn(self.input_layer(s))
        
        # 2) 각 은닉층 처리 및 활성화
        for hidden in self.hidden_layers:
            
            x = self.activation_fn(hidden(x))
            
        # 평균(mu) 계산(출력층 통과 시 activation 함수 적용 안함) --> 행동의 환경 범위가 고정되있거나([-2,2]) 특정 범위로 제한되어 있으면(로봇 관절 회전 범위 -180도도~180도) 범위 제한 필요)
        mu = self.mu_layer(x)
        
        #로그 표준편차 쓰는 이유 :  N(μ,σ^2)에서 σ < 0 일 수 없음, 그런데 일반적인 신경망은 음수도 출력함, 그래서 exp를 취해서 양수로 만들어줌
        # 로그 표준편차 계산 후 tanh로 값 제약 ([-1, 1]) --> 이렇게 하는 이유는 행동의 범위를 -1~1 사이로 제한하기 위해서(환경마다 선택할 수 있는 행동의 최소값과, 최대값이 달라서, 먼저 -1~1사이로 맞추고 min-max정규화의 역산으로 원래 행동의 범위로 되돌리는 것이 더 좋기 때문이다)
        log_std = torch.tanh(self.log_std_layer(x)) # 음수 출력 가능(신경망이 그냥 출력하니까까)
        
        # 표준편차는 exp(log_std)로 얻음
        std = log_std.exp() #음수 출력 불가능
        
        return mu, std

    



# 이산 행동 공간을 가진 정책 네트워크
class MLPDiscretePolicy(nn.Module):
    
    def __init__(self, 
                 state_dim,                           # 상태 벡터의 차원 수 (입력 크기)
                 action_dim,                          # 행동의 수 (출력 크기)
                 hidden_dims=(512,),                 # 은닉층의 각 레이어 크기를 순서대로 나열한 튜플
                 activation_fn=F.relu):              # 은닉층에 사용할 활성화 함수 (기본: ReLU)

        # 부모 클래스의 생성자 호출
        super(MLPDiscretePolicy, self).__init__()
        
        # 입력층: 상태 차원(state_dim) → 첫 번째 은닉층 크기(hidden_dims[0])
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        
        # 추가 은닉층들을 ModuleList로 관리(명시적으로 신경망 선언하는게 아니라 for문처럼 동적으로 만들어버리고 싶을때 사용)
        self.hidden_layers = nn.ModuleList()
        
        # 은닉층 추가
        for i in range(len(hidden_dims) - 1):
            
            # hidden_dims[i] → hidden_dims[i+1] 형태의 선형 레이어 추가 ex) input→512, 512→action_dim, 만약 (512,514)였다면? input->512->514->action_dim
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            
            self.hidden_layers.append(hidden_layer)
        
        # 출력층: 마지막 은닉층 크기(hidden_dims[-1]) → 행동 수(action_dim)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # 활성화 함수 저장 (forward 시에 사용)
        self.activation_fn = activation_fn

    # 정책 네트워크의 순전파 과정
    def forward(self, s):
        
        # 1) 입력층 통과 후 활성화 적용
        x = self.activation_fn(self.input_layer(s))
        
        # 2) 각 은닉층 순차적으로 통과하며 활성화 적용
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        
        # 3) 출력층 통과 후 소프트맥스로 확률 분포 생성(출력층 통과 시 activation 함수 적용 안함, 바로 소프트맥스 적용)
        logits = self.output_layer(x)
        
        # 소프트맥스 함수를 적용하여 확률 분포 생성(dim= -1 )
        prob = F.softmax(logits, dim=-1)
        
        return prob
    
    
    
    
    
    
    
    
# Baseline REINFORCE 알고리즘 클래스 (Advantage 방식, 상태가치함수 포함)
class BaselineREINFORCE:
    def __init__(
        self,
        state_dim,      # 상태 벡터 차원
        action_dim,     # 행동 벡터 차원
        gamma=0.99,     # 할인율 (미래 보상 감가)
        policy_lr=0.0003,   # 정책 신경망 학습률
        value_lr=0.0003,    # 상태가치 신경망 학습률
        action_type='continuous'  # 행동공간 종류 ('continuous' 또는 'discrete')
    ):
        # CUDA(GPU) 사용 가능하면 GPU, 아니면 CPU 디바이스 선택
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 정책 신경망(MLPGaussianPolicy) 생성, 상태 → 행동분포(평균/표준편차) 반환
        self.policy = MLPGaussianPolicy(state_dim, action_dim).to(self.device)
        
        # 상태가치 신경망(MLPStateValue) 생성, 상태 → 스칼라 가치값 반환
        self.value = MLPStateValue(state_dim).to(self.device)
        
        # 행동 타입(연속/이산) 저장
        self.action_type = action_type
        
        # 할인율 저장
        self.gamma = gamma
        
        # 정책, 가치 신경망 각각의 Adam 옵티마이저 준비
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        # 한 에피소드 동안 데이터를 쌓아둘 버퍼
        self.buffer = []

    @torch.no_grad()  # 행동을 뽑을 때는 역전파 X, 메모리 절약, 속도 향상
    def act(self, s, training=True):
        # 네트워크를 학습 모드/평가 모드로 전환
        self.policy.train(training)
        
        # 상태를 텐서(float)로 변환하고 디바이스에 올림
        s = torch.as_tensor(s, dtype=torch.float, device=self.device)
        
        if self.action_type == 'discrete':
            # 이산 행동공간: softmax 확률벡터 반환
            prob = self.policy(s)
            # 학습 중이면 확률적으로 샘플링, 아니면 최대 확률 행동 선택
            a = torch.multinomial(prob, 1) if training else torch.argmax(prob, dim=-1, keepdim=True)
        else:
            # 연속 행동공간: 평균(mu), 표준편차(std) 반환
            mu, std = self.policy(s)
            # 학습 중이면 표준정규에서 noise를 더해 샘플링, 아니면 평균값 사용
            z = torch.normal(mu, std) if training else mu
            # 출력 범위를 (-1, 1)로 제한(tanh 함수)
            a = torch.tanh(z)

        # 텐서를 CPU로 옮기고 numpy로 반환
        return a.cpu().numpy()

    def learn(self):
        self.policy.train()  # 학습 모드
        
        # 버퍼에 쌓인 transition들을 s, a, r, s', done, truncated로 분리(각각 numpy array)
        s, a, r, _, _, _ = map(np.stack, zip(*self.buffer))
        # 텐서(float)로 변환하고 디바이스에 올림
        s, a, r = map(lambda x: torch.as_tensor(x, dtype=torch.float, device=self.device), [s, a, r])
        # 보상 shape 맞추기 (batch, 1)
        r = r.unsqueeze(1)

        # Return(누적 할인 보상) 계산: G_t = r_t + gamma*G_{t+1} (뒤에서부터 누적)
        ret = torch.clone(r)
        for t in reversed(range(len(ret) - 1)):
            ret[t] += self.gamma * ret[t + 1]

        # 행동 로그확률 계산
        if self.action_type == 'discrete':
            # softmax 정책에서 실제 행동 a의 확률만 골라 로그 취함
            probs = self.policy(s)
            log_probs = torch.log(probs.gather(1, a.long()))
        else:
            # 연속 정책은 Gaussian 분포에서 log_prob 계산
            mu, std = self.policy(s)
            m = Normal(mu, std)
            # tanh로 squashing된 a를 역함수(atanh)로 변환, clipping으로 극단값 방지
            z = torch.atanh(torch.clamp(a, -1.0 + 1e-7, 1.0 - 1e-7))
            log_probs = m.log_prob(z).sum(dim=-1, keepdim=True)

        # 상태가치 신경망에서 각 상태별 예측값 얻기
        v = self.value(s)

        # 정책 손실(Advantage 방식): -(log pi(a|s) * (G_t - V(s)) )의 평균
        # Advantage = 실제 Return - 예측가치
        policy_loss = -(log_probs * (ret - v.detach())).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 상태가치 신경망 손실(MSE): (예측값과 실제 return 차이의 제곱 오차)
        value_loss = F.mse_loss(v, ret)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 손실값 기록 및 반환
        result = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
        return result
    
    def process(self, transition):
        result = None
        self.buffer.append(transition)  # 버퍼에 transition 추가
        
        # done or truncated(시간 초과 등) 발생 시 학습 시작
        if transition[-1] or transition[-2]:
            result = self.learn()
            self.buffer = []  # 버퍼 초기화(에피소드 단위 학습)
        return result
