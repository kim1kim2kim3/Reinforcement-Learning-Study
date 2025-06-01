import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Normal



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
    
    
    
    
    
    
    
    
# REINFORCE 알고리즘을 구현한 클래스
class REINFORCE:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        
        # 사용할 디바이스를 설정: CUDA 가능하면 GPU, 아니면 CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 상태 차원(state_dim)과 행동 차원(action_dim)에 맞춰 정책 신경망을 초기화하고 디바이스에 올림림
        self.policy = MLPDiscretePolicy(state_dim, action_dim).to(self.device)
        
        # 할인율 gamma 설정
        self.gamma = gamma
        
        # Adam 옵티마이저로 정책 네트워크 파라미터를 학습률 lr로 업데이트할 준비
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # 에피소드 동안 저장할 경험 버퍼 초기화
        self.buffer = []
        
        
        
    @torch.no_grad()  #행동 선택 시 기울기 계산 방지
    def act(self, s, training=True):  # 행동 선택 함수
        
        # 네트워크를 train 모드(training=True) 또는 eval 모드(training=False)로 설정(파이토치에서 내부의 모드 설정)
        self.policy.train(training)

        # 넘파이 배열 s를 텐서로 변환하고 디바이스에 올림
        s = torch.as_tensor(s, dtype=torch.float, device=self.device)
        
        # 정책 네트워크로부터 행동 확률 분포를 얻음
        prob = self.policy(s)
        
        # 학습 중(training=True)이면 multinomial 샘플링, 아니면 확률 최대 행동 선택    X if 조건 else Y
        a = torch.multinomial(prob, 1) if training else torch.argmax(prob, dim=-1, keepdim=True)

        # 행동 텐서를 CPU로 옮겨 넘파이 배열로 반환
        return a.cpu().numpy()



    def learn(self):
        
        # 학습 모드로 전환
        self.policy.train()
        
        # 버퍼에 쌓인 transition들을 numpy 배열로 분리: s, a, r, 다음 s, done, truncated
        s, a, r, _, _, _ = map(np.stack, zip(*self.buffer))
        
        # 상태, 행동, 보상을 텐서로 변환하고 디바이스에 올린다이
        s, a, r = map(lambda x: torch.as_tensor(x, dtype=torch.float, device=self.device), [s, a, r])
        
        # 행동 텐서를 long 타입으로 변환해 인덱싱에 사용 가능하게 한다이
        a = a.long()
        
        # 보상 차원 맞추려고 (batch, 1) 모양으로 변환
        r = r.unsqueeze(1)
        
        # 리턴값 G_t 계산: 뒤에서부터 누적 할인 보상 더한다이
        ret = torch.clone(r)
        for t in reversed(range(len(ret) - 1)):
            ret[t] += self.gamma * ret[t + 1]
            
        # 상태들에 대한 정책 확률 얻기
        probs = self.policy(s)
        
        # 실제 선택된 행동 a의 확률만 뽑아서 로그 취함
        log_probs = torch.log(probs.gather(1, a))
        
        # REINFORCE 손실: - E[G_t * log π(a|s)]
        policy_loss = - (ret * log_probs).mean() # 원래는 그냥 sum() 해도 되는데 평균을 취해서 더 안정적으로 학습
        
        # 옵티마이저 기울기 초기화
        self.optimizer.zero_grad()
        
        # 손실 역전파
        policy_loss.backward()
        
        # 파라미터 업데이트
        self.optimizer.step()
        
        # 학습 결과로 손실 값 반환
        result = {'policy_loss': policy_loss.item()}
        return result
    
    
    # 경험 버퍼에 경험 추가
    def process(self, transition):
        result = None
        
        # transition = (state, action, reward, next_state, done, truncated)
        self.buffer.append(transition)
        
        # 에피소드 끝이면 학습 진행
        if transition[-1] or transition[-2]:
            
            result = self.learn()
            
            # 버퍼 비우기
            self.buffer = []
        return result
