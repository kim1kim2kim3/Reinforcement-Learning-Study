# a2c_main_pytorch.py
# PyTorch 버전 A2C main 예시
import gym
import torch
from a2c_learn import A2CAgent  # 이전에 변환해 둔 PyTorch A2C 코드를 담은 파일

def main():
    # 환경 이름: TF 코드와 동일하게 'Pendulum-v0'을 쓰되,
    # 만약 로컬에 v0이 없으면 'Pendulum-v1'으로 바꿔도 됩니다.
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    # 에이전트 생성
    agent = A2CAgent(env)

    # Actor 네트워크 가중치 로드
    # - TF 버전은 "./save_weights/pendulum_actor.h5"였지만,
    #   PyTorch 예시에서는 "./save_weights/pendulum_actor.pth"로 가정합니다.
    actor_weights_path = "./save_weights/pendulum_actor.pth"
    agent.actor.load_state_dict(torch.load(actor_weights_path, map_location=agent.device))
    agent.actor.eval()  # 평가 모드로 전환 (드롭아웃, 배치정규화 등이 있으면 영향 없도록)

    state = env.reset()
    if isinstance(state, tuple):
        # gymnasium 환경일 경우 (obs, info) 튜플 반환 가능
        state = state[0]

    time_step = 0
    done = False

    while True:
        env.render()

        # 상태를 텐서로 변환 [1, state_dim]
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(agent.device)

        # Actor forward: mu, std 반환
        with torch.no_grad():
            mu, std = agent.actor(state_tensor)
        # mu는 [1, action_dim] 형태, numpy로 바꿔서 첫 번째 배치만 사용
        action = mu.cpu().numpy().squeeze(0)  # [action_dim]

        # 환경에 action 적용
        next_step = env.step(action)
        if len(next_step) == 5:
            # (obs, reward, terminated, truncated, info) 형태
            next_state, reward, terminated, truncated, info = next_step
            done = terminated or truncated
        else:
            # (obs, reward, done, info) 형태
            next_state, reward, done, info = next_step

        time_step += 1
        print(f"Time: {time_step}, Reward: {reward:.2f}")

        if done:
            break

        # 상태 갱신
        state = next_state if not isinstance(next_state, tuple) else next_state[0]

    env.close()


if __name__ == "__main__":
    # numpy를 import하지 않았다면 추가
    import numpy as np
    main()