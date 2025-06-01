#그냥 gymnasium 관련 튜토리얼 코드

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

from torch.distributions import Normal

# 환경 불러오기기
env = gym.make('CartPole-v1')

print("---------------------------------------------------------------")

# 환경 초기화 하기기(보통 환경 초기화는 랜덤성을 포함하고 있어 반환하는 초기 상태의 값이 달라짐짐)
s, info = env.reset()

print("Initial state is: ", s)
print("Information: ", info)

print("---------------------------------------------------------------")

# 행동 공간에서 임의의 행동 하나를 가져오자자
a = env.action_space.sample()  

# env.step(action)을 통해서 환경에 행동을 취하자자
s_prime, r, terminated, truncated, info =  env.step(a)

print("selected action:", a)         # 선택된 행동
print("Next state: ", s_prime)       # 다음 상태
print("Reward: ", r)                 # 보상
print("Is terminated? ", terminated) # 환경이 종료 조건에 의해 종료되었는지 여부(ex) 에이전트가 목적을 달성했거나 돌이길 수 없는 상태에 빠진 경우)
print("Is truncated? ", truncated)   # 환경이 최대 상호작용 횟수에 도달하여 종료되었는지 여부(환경의 종료 조건이 따로 없는 경우에만 의미 있음 ex) 로봇 통제와 같은 무한히 상호작용할 수 있는 환경의 경우 1000회 정도 상호작용 후 환경을 마침)
print("Information: ", info)         # 추가 정보


print("---------------------------------------------------------------")

# 위의 내용을 종합한 한 에피소드 진행

s, terminated, truncated, ret = env.reset(), False, False, 0
while not (terminated or truncated):
    a = env.action_space.sample()
    s_prime, r, terminated, truncated, _ = env.step(a)
    
    ret += r
    s = s_prime
print("Return: ", ret)


print("---------------------------------------------------------------")

# seed 추가 ver

seed = 0
env = gym.make('CartPole-v1')

s, info = env.reset(seed=seed)
print(s)

s, info = env.reset()
print(s)

s, info = env.reset()
print(s)

s, info = env.reset(seed=seed)
print(s)

s, info = env.reset()
print(s)



print("---------------------------------------------------------------")