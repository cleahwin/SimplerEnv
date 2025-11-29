#!/bin/bash

# for carrot task
# scene_name=bridge_table_1_v1
# robot=widowx
# rgb_overlay_path=/home/cleah/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=/home/cleah/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png

robot_init_x=0.127
robot_init_y=0.06
# env_name=PutCarrotOnPlateInScene-v0
env_name=PutEggplantInBasketScene-v0

sim_freq=500
control_freq=5
max_episode_steps=120

obj_episode_range_start=0
obj_episode_range_end=24

python simplerenv_server.py \
  --robot ${robot} \
  --control-freq ${control_freq} \
  --sim-freq ${sim_freq} \
  --max-episode-steps ${max_episode_steps} \
  --env-name ${env_name} \
  --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  # --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
  # --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
  # --robot-init-rot-quat-center 0 0 0 1 \
  # --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --port 5000
