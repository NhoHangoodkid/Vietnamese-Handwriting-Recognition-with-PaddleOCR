from pathlib import Path
import os
import yaml

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_WEIGHT_PATH = WEIGHTS_DIR / "weight_svtr.pdparams"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "svtr_config.yml"
DEFAULT_CHARSET_PATH = PROJECT_ROOT / "charset_official_ver3.txt"
PADDLEOCR_DIR = PROJECT_ROOT / "PaddleOCR"
SAMPLE_DIR = PROJECT_ROOT / "samples"

class AppConfig:
    """Application Configuration Settings"""
    def __init__(self, config_file: str = None):
        self.root_dir = PROJECT_ROOT
        self.config_path = Path(config_file) if config_file else DEFAULT_CONFIG_PATH
        self.weight_path = DEFAULT_WEIGHT_PATH
        self.charset_path = DEFAULT_CHARSET_PATH
        self.use_gpu = False
        self.language = "vi"
        
        # Load from YAML if available
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.yaml_data = yaml.safe_load(f)
            except Exception:
                self.yaml_data = {}
        else:
            self.yaml_data = {}

    def get_charset_list(self):
        """Read list of supported characters"""
        if not self.charset_path.exists():
            return []
        with open(self.charset_path, "r", encoding="utf-8") as f:
            chars = [line.strip("\r\n") for line in f if line.strip("\r\n")]
        return chars

    def check_environment(self):
        """Check status of required files"""
        status = {
            "weights_exist": self.weight_path.exists(),
            "config_exist": self.config_path.exists(),
            "charset_exist": self.charset_path.exists(),
            "paddleocr_submodule": (PADDLEOCR_DIR / "tools" / "infer_rec.py").exists(),
        }
        return status
