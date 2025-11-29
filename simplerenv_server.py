#!/usr/bin/env python
import asyncio
import websockets
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import argparse

from simpler_env.utils.env.env_builder import build_maniskill2_env

IMG_SIZE = (256, 256)

def to_python_scalar(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.int8, np.int16, np.int32, np.int64)):
        return int(x)
    if isinstance(x, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(x)
    if isinstance(x, (np.float16, np.float32, np.float64)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (bool, int, float, str, type(None))):
        return x
    try:
        return str(x)
    except Exception:
        return None

def convert_dict_to_python(d):
    if isinstance(d, dict):
        result = {}
        for k, v in d.items():
            try:
                converted = convert_dict_to_python(v)
                json.dumps(converted)
                result[k] = converted
            except (TypeError, ValueError):
                try:
                    result[k] = str(v)
                except Exception:
                    print(f"Warning: Skipping non-serializable key '{k}'")
        return result

    elif isinstance(d, (list, tuple)):
        result = []
        for item in d:
            try:
                converted = convert_dict_to_python(item)
                json.dumps(converted)
                result.append(converted)
            except (TypeError, ValueError):
                try:
                    result.append(str(item))
                except Exception:
                    print(f"Warning: Skipping non-serializable list item")
        return result

    else:
        try:
            converted = to_python_scalar(d)
            json.dumps(converted)
            return converted
        except (TypeError, ValueError):
            return str(d)

class SimplerEnvServer:
    def __init__(self, args):
        self.args = args
        self.host = "0.0.0.0"
        self.port = args.port
        self.env = None
        self.policy_client = None

    def create_env(self):
        env_kwargs = dict(
            obs_mode="rgbd",
            robot=self.args.robot,
            rgb_overlay_path=self.args.rgb_overlay_path,
            rgb_overlay_cameras=["3rd_view_camera"],
            sim_freq=self.args.sim_freq,
            control_freq=self.args.control_freq,
            max_episode_steps=self.args.max_episode_steps,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            camera_cfgs={"add_segmentation": True},
        )

        if self.args.scene_name is not None:
            env_kwargs["scene_name"] = self.args.scene_name

        print("Creating environment with:", env_kwargs)

        self.env = build_maniskill2_env(self.args.env_name, **env_kwargs)
        print(f"Environment {self.args.env_name} created")

    def encode_image(self, obs):
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']['3rd_view_camera']['rgb']
        elif isinstance(obs, np.ndarray):
            img = obs
        else:
            img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)

        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _convert_info(self, info):
        info_clean = {}
        for key, value in info.items():
            try:
                converted = convert_dict_to_python(value)
                info_clean[key] = converted
            except Exception:
                info_clean[key] = str(value)
        return info_clean

    async def handle_client(self, websocket):
        print(f"Client connected from {websocket.remote_address}")
        try:
            async for message in websocket:
                data = json.loads(message)
                command = data["command"]

                if command == "create_env":
                    self.create_env()
                    self.policy_client = WebsocketClientPolicy(host=self.host, port=self.port)
                    await websocket.send(json.dumps({"status": "ok"}))

                elif command == "reset":
                    task_description = data.get("task_description", "")
                    if task_description != self.current_task:
                        self.current_task = task_description
                        self.is_reset = True
                    
                    obs, info = self.env.reset()
                    
                    # Prepare observation for policy
                    policy_obs = {
                        "image": obs['image']['3rd_view_camera']['rgb'],
                        "eef_pos": info.get('eef_pos', np.zeros(8)),  # Default to zeros if not available
                        "task_description": self.current_task
                    }
                    
                    # Get initial action from policy
                    if self.policy_client:
                        policy_action = self.policy_client.infer(policy_obs)
                        action = self._process_policy_action(policy_action)
                        self.is_reset = False
                    else:
                        action = np.zeros(self.env.action_space.shape)
                    
                    response = {
                        "observation": self.encode_image(obs),
                        "info": self._convert_info(info),
                        "action": action.tolist()  # Send initial action
                    }
                    await websocket.send(json.dumps(response))
                elif command == "step":
                    # Get action from client or use policy
                    if self.policy_client and not self.is_reset:
                        # Get current observation
                        obs = self.env.get_obs()
                        policy_obs = {
                            "image": obs['image']['3rd_view_camera']['rgb'],
                            "eef_pos": obs.get('eef_pos', np.zeros(8)),
                            "task_description": self.current_task
                        }
                        policy_action = self.policy_client.infer(policy_obs)
                        action = self._process_policy_action(policy_action)
                    else:
                        action = np.array(data.get("action", []), dtype=np.float32)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    response = {
                        "observation": self.encode_image(obs),
                        "reward": to_python_scalar(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": self._convert_info(info),
                        "action": action.tolist()  # Return the action that was used
                    }
                    await websocket.send(json.dumps(response))


                elif command == "close":
                    await websocket.send(json.dumps({"status": "closing"}))
                    break

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print("Error:", e)
            import traceback; traceback.print_exc()
            await websocket.send(json.dumps({"error": str(e)}))

async def main(args):
    server = SimplerEnvServer(args)
    print(f"Server running on port {args.port}")
    async with websockets.serve(server.handle_client, "0.0.0.0", args.port, ping_interval=None, ping_timeout=None, close_timeout=10):
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--robot", type=str, required=True)
    parser.add_argument("--scene-name", type=str)

    parser.add_argument("--rgb-overlay-path", type=str)
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=120)

    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()
    asyncio.run(main(args))
