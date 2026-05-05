"""
Single-episode debug: trace what actually happens step by step.
Run this to find why LCO never succeeds.
"""
import sys, os
sys.path.insert(0, r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Scripts")

import numpy as np
import torch

# Import the env and simulator directly
from charging_env import ChargingEnvV3
from pack_simulator import BatteryPackSimulator

print("="*60)
print("Single-episode trace — LCO at max current")
print("="*60)

# Create LCO env
env = ChargingEnvV3('LCO')
obs, _ = env.reset()
print(f"Initial obs: {obs}")
print(f"Initial SOC_min: {obs[1]:.4f}")

# Run with fixed max action (I_pack = 45A)
action = np.array([1.0])  # max action -> tanh(1.0)*22.25+22.75 = I_pack max

total_r = 0
for step in range(10):
    obs, r, done, trunc, info = env.step(action)
    total_r += r
    soc_min = obs[1]
    print(f"  step={step+1:3d} | action={action[0]:.3f} | I_pack={info.get('I_pack',0):.2f}A | "
          f"SOC_min={soc_min:.4f} | r={r:.4f} | done={done} | reason={info.get('done_reason','')}")
    if done or trunc:
        print(f"  TERMINATED: {info.get('done_reason','?')}")
        break

print(f"\nCheck action -> I_pack mapping:")
for a in [0.5, 0.9, 0.99, 1.0]:
    # replicate ppo_train action mapping
    I_pack = a * 22.25 + 22.75  # or however it's done
    print(f"  action={a:.2f} -> I_pack={I_pack:.2f}A -> I_cell={I_pack/9:.3f}A")