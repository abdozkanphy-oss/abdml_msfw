
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

class ConfigReader:
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        with open(f"{BASE_DIR}/config.json", "r") as config_dict:
            self.__dict__ = json.load(config_dict)
        
    def __getitem__(self, key):
        return getattr(self, key)

