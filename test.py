from data import get_data_loader
import yaml
from visualize import make_animation
import torch

config_path = "config.yaml"

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
train_loader, valid_loader, test_loader = get_data_loader(config)


