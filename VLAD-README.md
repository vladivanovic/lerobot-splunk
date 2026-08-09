# Create isolated environment — avoids conflicts with system Python
conda create -n lerobot_gpu python=3.10 -y
conda activate lerobot_gpu

# Check your JetPack version first
sudo apt-cache show nvidia-jetpack | grep Version

# Download the correct PyTorch wheel for your JetPack version from:
# https://developer.download.nvidia.com/compute/redist/jp/

# Example install pattern (verify exact URL for your JetPack version):
pip install torch-*.whl torchvision-*.whl

cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install with feetech motor support (SO101 uses Feetech servos)
pip install -e ".[feetech]"

# Verify install
python3 -c "import lerobot; print(lerobot.__version__)"

sudo pip3 install jetson-stats
sudo systemctl enable jtop.service
sudo systemctl start jtop.service

# Also install inside your conda env
conda activate lerobot_gpu
pip install jetson-stats

sudo usermod -aG dialout $USER
# Log out and back in for group change to take effect

# Or temporarily:
sudo chmod 666 /dev/ttyACM0

# List available cameras
python3 -c "
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
print(OpenCVCamera.find_cameras())
"

# Or manually check devices
ls /dev/video*
v4l2-ctl --list-devices

python3 -c "
import cv2
cap = cv2.VideoCapture(2)  # adjust index
ret, frame = cap.read()
print('Camera OK' if ret else 'Camera FAILED', frame.shape if ret else '')
cap.release()
"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 360, fps: 30, fourcc: MJPG}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm \
  --display_data=true \
  --dataset.repo_id=YOUR_HF_USERNAME/lego-block-front-only \
  --dataset.single_task="Put lego brick into the plate box" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=15

  
# Check episode count and stats
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('YOUR_HF_USERNAME/lego-block-front-only')
print(f'Episodes: {ds.num_episodes}')
print(f'Total frames: {len(ds)}')
print(f'Features: {list(ds.features.keys())}')
"

huggingface-cli login  # one-time auth

# push_to_hub defaults to true in lerobot-record, but if you disabled it:
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('YOUR_HF_USERNAME/lego-block-front-only')
ds.push_to_hub()
"

lerobot-train \
  --dataset.repo_id=YOUR_HF_USERNAME/lego-block-front-only \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/lego-block-front-only \
  --job_name=lego-block-front-only \
  --steps=120000 \
  --batch_size=2 \
  --save_freq=10000 \
  --eval_freq=20000 \
  --wandb.enable=false

  # If wandb is enabled
# Visit https://wandb.ai/your-project

# Otherwise check logs directly
tail -f outputs/train/lego-block-front-only/train.log


lerobot-train \
  --config_path=outputs/train/lego-block-front-only/checkpoints/last/pretrained_model/train_config.json \
  --resume=true

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 360, fps: 30, fourcc: MJPG}}" \
  --display_data=false \
  --dataset.repo_id=YOUR_HF_USERNAME/eval_lego-block-front-only \
  --dataset.single_task="Put lego brick into the plate box" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=15 \
  --policy.path=outputs/train/lego-block-front-only/checkpoints/last/pretrained_model

python3 LeRobot-to-Splunk.py

