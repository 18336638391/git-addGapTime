import os
import time
import datetime
import subprocess
from collections import deque
import torch
from deepseek.nlp import DeepSeekLMHeadModel, DeepSeekTokenizer
from deepseek.hai.trainer import HAITrainer, TrainingArguments
from deepseek.hai.dataset import DataCollatorForCausalLM
import deepspeed
from peft import LoraConfig, get_peft_model
import threading
import traceback
import sys
import speech_recognition as sr
import pyttsx3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ThreadPoolExecutor
import requests
import cv2
import re
from PIL import Image
from PIL import ImageOps
import numpy as np

nltk.download('all', quiet=True)

# ========== 全局配置 ==========
class Config:
    APP_NAME = "GapTimeLLMS-Pro"
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    TEMP_DIR = os.path.join(os.getcwd(), "temp")
    MAX_RESTARTS = 10           
    RESTART_DELAY = 60          
    EVENT_HISTORY_SIZE = 1000   
    LLM_MODEL = "deepseek-617b-base"  
    TIME_DIM = 256              
    FEEDBACK_LOG = os.path.join(LOG_DIR, "user_feedback.log")
    UPGRADE_TRIGGER_INTERVAL = 86400 
    CODE_TEMPLATE_PATH = "gap_time_template.py"
    COMMAND_TIMEOUT = 30        
    TRAINING_DATA_PATH = os.path.join(TEMP_DIR, "training_data.csv")
    LEARNING_RATE = 1e-5        
    TRAINING_EPOCHS = 2         
    MIN_TRAINING_SAMPLES = 200   
    AUTO_TRAIN_INTERVAL = 1800   
    SPEECH_RECOGNITION_LANGUAGE = "zh-CN"
    DATA_SCALING = True
    VISUALIZATION_DIR = os.path.join(TEMP_DIR, "visualizations")
    PRE_TRAINING_DATA_PATH = os.path.join(TEMP_DIR, "pre_training_data.csv")
    MAX_THREADS = 20            
    DEVICE_MONITOR_INTERVAL = 30 
    ERROR_RECOVERY_DELAY = 60   
    ENABLE_SPEECH_OUTPUT = True
    ALLOWED_LIBS = [
        "opencv-python", "requests", "matplotlib", "torch",
        "deepseek-hai", "deepseek-nlp", "deepspeed", "peft",
        "pyttsx3", "speechrecognition", "pillow", "scipy",
        "sounddevice", "torchvision"
    ]
    DISTRIBUTED_WORKERS = 8     
    CRAWLER_THREADS = 5         
    SECURITY_PATTERNS = [
        r'\brm\b|\bkill\b|\bchmod\b', r'^(sudo|su)\b',
        r'\beval\b|\bexec\b|\bimport\b', r'\bdelete\b|\bformat\b'
    ]
    MODEL_SAVE_INTERVAL = 1800  
    AUTO_OPTIMIZE = True        
    MULTIMODAL_THRESHOLD = 0.8  
    SELF_ITERATION_STEPS = 1000  
    DEVICE_THRESHOLD = {
        "GPU_MEMORY": 30,
        "CPU_LOAD": 8.0
    }

# ========== 增强型日志系统 ==========
class Logger:
    def __init__(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        self.log_file = os.path.join(Config.LOG_DIR, f"{Config.APP_NAME}_{time.strftime('%Y%m%d')}.log")
        self.lock = threading.Lock()
        self.resource_monitor = ResourceMonitor(self)
        self.resource_monitor.start()

    def log(self, message: str, level="INFO"):
        with self.lock:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}"
            print(log_entry)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")

# ========== 智能设备检测 ==========
class DeviceDetector:
    @staticmethod
    def detect_camera():
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return True
        return False

    @staticmethod
    def detect_audio_device():
        try:
            import sounddevice as sd
            return len(sd.query_devices()) > 0
        except:
            return False

# ========== 资源监控模块 ==========
class ResourceMonitor(threading.Thread):
    def __init__(self, logger):
        super().__init__(daemon=True)
        self.logger = logger
        self.running = True
        self.prev_gpu_memory = 0

    def run(self):
        while self.running:
            self.monitor_resources()
            time.sleep(Config.DEVICE_MONITOR_INTERVAL)

    def monitor_resources(self):
        try:
            if torch.cuda.is_available():
                current_gpu = deepspeed.runtime.utils.get_current_device_memory_used() / 1024**3
                self.logger.log(f"GPU内存: {current_gpu:.2f} GB (变化: {current_gpu - self.prev_gpu_memory:.2f} GB)", "DEBUG")
                self.prev_gpu_memory = current_gpu
            
            cpu_load = os.getloadavg()[0]
            self.logger.log(f"CPU负载: {cpu_load:.2f}", "DEBUG")
            
            self.auto_optimize()
        except Exception as e:
            self.logger.log(f"资源监控错误: {str(e)}", "ERROR")

    def auto_optimize(self):
        if torch.cuda.is_available():
            mem_used = deepspeed.runtime.utils.get_current_device_memory_used() / 1024**3
            if mem_used > Config.DEVICE_THRESHOLD["GPU_MEMORY"]:
                torch.cuda.empty_cache()
                Config.MAX_THREADS = 8
            else:
                Config.MAX_THREADS = 20
        
        if os.getloadavg()[0] > Config.DEVICE_THRESHOLD["CPU_LOAD"]:
            Config.AUTO_TRAIN_INTERVAL = max(Config.AUTO_TRAIN_INTERVAL + 300, 1800)

# ========== 安全扫描模块 ==========
class AdvancedSecurityScanner:
    @staticmethod
    def scan_command(command):
        return not re.search("|".join(Config.SECURITY_PATTERNS), command, re.IGNORECASE)

    @staticmethod
    def is_safe_library(lib):
        return lib in Config.ALLOWED_LIBS

# ========== 大语言模型核心 ==========
class GapTimeLLM:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DeepSeekLMHeadModel.from_pretrained(
            Config.LLM_MODEL,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True,
            use_flash_attn=True
        )
        
        self.model, self.optimizer, _ = deepspeed.initialize(
            model=self.model,
            config_file="deepspeed_617b.json",
            model_parameters=self.model.named_parameters(),
            optimizer=torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        )
        
        self.tokenizer = DeepSeekTokenizer.from_pretrained(Config.LLM_MODEL)
        self.lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["query_key_value"])
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.eval()
        self.thinking = False

    def generate_response(self, prompt: str, max_length=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs, 
                max_length=max_length, 
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def deep_reasoning(self, prompt, steps=10):
        self.thinking = True
        for step in range(steps):
            prompt = self.generate_response(prompt, temperature=0.9 - 0.05*step)
            self.logger.log(f"思考步骤{step+1}: {prompt[:50]}", "DEBUG")
        self.thinking = False
        return prompt

# ========== 训练引擎 ==========
class TrainingEngine:
    def __init__(self, logger: Logger, llm: GapTimeLLM):
        self.logger = logger
        self.llm = llm
        self.data_buffer = deque(maxlen=1000)
        self.load_pretrained_data()

    def load_pretrained_data(self):
        if os.path.exists(Config.PRE_TRAINING_DATA_PATH):
            with open(Config.PRE_TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    self.data_buffer.append(tuple(line.strip().split("\t", 1)))
            self.logger.log(f"加载{len(self.data_buffer)}条预训练数据", "INFO")

    def add_training_sample(self, prompt, response):
        self.data_buffer.append((prompt, response))
        self.logger.log(f"新增训练样本，当前队列大小: {len(self.data_buffer)}", "DEBUG")
        self.trigger_training()

    def trigger_training(self):
        if len(self.data_buffer) >= Config.MIN_TRAINING_SAMPLES:
            self.run_distributed_training()

    def run_distributed_training(self):
        training_args = TrainingArguments(
            output_dir=Config.TEMP_DIR,
            num_train_epochs=Config.TRAINING_EPOCHS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=Config.LEARNING_RATE,
        )
        
        data_collator = DataCollatorForCausalLM(
            tokenizer=self.llm.tokenizer,
            mlm=False,
            padding="max_length"
        )
        
        trainer = HAITrainer(
            model=self.llm.model,
            args=training_args,
            train_dataset=list(self.data_buffer),
            data_collator=data_collator
        )
        
        trainer.train()
        self.llm.model.eval()
        self.logger.log("分布式训练完成", "INFO")

# ========== 自主执行核心 ==========
class AutonomousCore:
    def __init__(self, logger: Logger, training_engine: TrainingEngine):
        self.logger = logger
        self.training_engine = training_engine
        self.is_autonomous = False
        self.mode = "manual"
        self.iteration_counter = 0
        self.initiate_self_diagnostics()

    def initiate_self_diagnostics(self):
        threading.Thread(target=self.run_diagnostics, daemon=True).start()

    def run_diagnostics(self):
        try:
            self.check_model_health()
            self.check_data_integrity()
        except Exception as e:
            self.logger.log(f"诊断错误: {str(e)}", "ERROR")

    def check_model_health(self):
        self.training_engine.llm.generate_response("诊断测试: 模型是否正常工作")
        self.logger.log("模型健康检查通过", "INFO")

    def check_data_integrity(self):
        if os.path.exists(Config.TRAINING_DATA_PATH):
            self.logger.log(f"数据文件存在，路径: {Config.TRAINING_DATA_PATH}", "INFO")

    def toggle_mode(self, new_mode):
        if new_mode in ["manual", "autonomous", "self-iterate"]:
            self.mode = new_mode
            self.logger.log(f"切换至{new_mode}模式", "INFO")
            if new_mode == "autonomous":
                self.start_autonomous_learning()

    def start_autonomous_learning(self):
        self.is_autonomous = True
        while self.is_autonomous:
            self.collect_environment_data()
            self.training_engine.trigger_training()
            time.sleep(Config.AUTO_TRAIN_INTERVAL)

    def collect_environment_data(self):
        if DeviceDetector.detect_camera():
            self.collect_visual_data()
        if DeviceDetector.detect_audio_device():
            self.collect_audio_data()

    def collect_visual_data(self):
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(Config.TEMP_DIR, "webcam_feed.jpg"), frame)
                cap.release()
        except Exception as e:
            self.logger.log(f"视觉数据采集失败: {str(e)}", "ERROR")

    def collect_audio_data(self):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source, timeout=5)
                with open(os.path.join(Config.TEMP_DIR, "mic_input.wav"), "wb") as f:
                    f.write(audio.get_wav_data())
        except Exception as e:
            self.logger.log(f"音频数据采集失败: {str(e)}", "ERROR")

# ========== 多模态执行器 ==========
class AdvancedMultiModalExecutor:
    def __init__(self, logger: Logger, llm: GapTimeLLM):
        self.logger = logger
        self.llm = llm
        self.vision_model = self.load_vision_model()

    def load_vision_model(self):
        try:
            from torchvision.models import resnet50
            return resnet50(pretrained=True).eval()
        except Exception as e:
            self.logger.log(f"加载视觉模型失败: {str(e)}", "ERROR")
            return None

    def handle_visual_input(self, path="webcam_feed.jpg"):
        if not self.vision_model or not os.path.exists(path):
            return "无法处理视觉输入"
        
        try:
            image = ImageOps.fit(Image.open(path), (224, 224))
            tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                output = self.vision_model(tensor)
            return f"图像分类结果: {torch.argmax(output).item()}"
        except Exception as e:
            self.logger.log(f"图像分析失败: {str(e)}", "ERROR")
            return "图像分析失败"

# ========== 主控控制器 ==========
class GapTimeController:
    def __init__(self):
        self.logger = Logger()
        self.llm = GapTimeLLM(self.logger)
        self.training_engine = TrainingEngine(self.logger, self.llm)
        self.autonomous_core = AutonomousCore(self.logger, self.training_engine)
        self.executor = AdvancedMultiModalExecutor(self.logger, self.llm)
        self.running = False

    def start(self):
        self.running = True
        self.autonomous_core.toggle_mode("autonomous")
        self.monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        self.monitor_thread.start()

    def monitor_system(self):
        while self.running:
            if self.autonomous_core.mode == "autonomous":
                self.autonomous_core.collect_environment_data()
            time.sleep(5)

    def shutdown(self):
        self.running = False
        self.autonomous_core.is_autonomous = False
        self.logger.log("程序安全关闭", "INFO")
        sys.exit(0)

# ========== 程序入口 ==========
if __name__ == "__main__":
    # 依赖安装检查
    required_packages = [
        "torch", "deepspeed", "deepseek-hai", "deepseek-nlp",
        "peft", "pillow", "scipy", "sounddevice", "torchvision"
    ]
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
    controller = GapTimeController()
    controller.logger.log(f"{Config.APP_NAME} 启动 (设备: {controller.llm.device})", "INFO")
    controller.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.shutdown()
