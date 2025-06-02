# a2c_main.py

from a2c_learn import A2CAgent  # 클래스 이름이 정확한지 확인
import gym

def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'      # v0 대신 v1을 권장합니다.
    env = gym.make(env_name)

    agent = A2CAgent(env)         # 생성자 호출은 그대로

    agent.train(max_episode_num)  # train 메서드 호출

    agent.plot_result()           # 결과 그리기

if __name__ == "__main__":
    main()