import os
import time
import datetime
import subprocess
from collections import deque
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
try:
    import cv2  # 视觉识别
except ImportError:
    cv2 = None
try:
    import requests  # 网络请求
    from concurrent.futures import ThreadPoolExecutor  # 分布式处理
except ImportError:
    requests = None
    ThreadPoolExecutor = None
import re  # 正则表达式安全扫描

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ========== 全局配置 ==========
class Config:
    APP_NAME = "GapTimeLLMS"
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    TEMP_DIR = os.path.join(os.getcwd(), "temp")
    MAX_RESTARTS = 5
    RESTART_DELAY = 10
    EVENT_HISTORY_SIZE = 200
    LLM_MODEL = "gpt2-medium"
    TIME_DIM = 128
    FEEDBACK_LOG = os.path.join(LOG_DIR, "user_feedback.log")
    UPGRADE_TRIGGER_INTERVAL = 3600
    CODE_TEMPLATE_PATH = "gap_time_template.py"
    COMMAND_TIMEOUT = 20
    TRAINING_DATA_PATH = os.path.join(TEMP_DIR, "training_data.csv")
    LEARNING_RATE = 5e-5
    TRAINING_EPOCHS = 15
    MIN_TRAINING_SAMPLES = 50
    AUTO_TRAIN_INTERVAL = 1800
    SPEECH_RECOGNITION_LANGUAGE = "zh-CN"
    DATA_SCALING = True
    VISUALIZATION_DIR = os.path.join(TEMP_DIR, "visualizations")
    PRE_TRAINING_DATA_PATH = os.path.join(TEMP_DIR, "pre_training_data.csv")
    MAX_THREADS = 10
    DEVICE_MONITOR_INTERVAL = 60
    ERROR_RECOVERY_DELAY = 30
    ENABLE_SPEECH_OUTPUT = True
    ALLOWED_LIBS = ["opencv-python", "requests", "matplotlib", "torch"]  # 安全库列表
    DISTRIBUTED_WORKERS = 4  # 分布式训练线程数
    CRAWLER_THREADS = 3  # 爬虫线程数
    SECURITY_PATTERNS = [r'\brm\b|\bkill\b|\bchmod\b', r'^(sudo|su)\b']  # 危险命令模式
    MODEL_SAVE_INTERVAL = 3600  # 自动保存间隔

# ========== 资源监控模块 ==========
class ResourceMonitor(threading.Thread):
    def __init__(self, logger):
        super().__init__(daemon=True)
        self.logger = logger
        self.running = True

    def run(self):
        while self.running:
            try:
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1024**3
                    self.logger.log(f"GPU内存使用: {mem:.2f} GB", "DEBUG")
                # 可扩展CPU/内存监控
            except Exception as e:
                self.logger.log(f"资源监控错误: {str(e)}", "ERROR")
            time.sleep(Config.DEVICE_MONITOR_INTERVAL)

    def stop(self):
        self.running = False

# ========== 安全扫描器 ==========
class SecurityScanner:
    @staticmethod
    def scan(command):
        return not re.search("|".join(Config.SECURITY_PATTERNS), command, re.IGNORECASE)

# ========== 增强型日志系统 ==========
class Logger:
    def __init__(self):
        self.log_file = os.path.join(Config.LOG_DIR, f"{Config.APP_NAME}_{time.strftime('%Y%m%d')}.log")
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        self.lock = threading.Lock()
        self.monitor = ResourceMonitor(self)
        self.monitor.start()

    def log(self, message: str, level="INFO"):
        with self.lock:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}"
            print(log_entry)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")

# ========== 大语言模型核心 ==========
class GapTimeLLM:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GPT2LMHeadModel.from_pretrained(Config.LLM_MODEL).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(Config.LLM_MODEL)
        self.model.eval()
        self.last_train_time = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scaler = MinMaxScaler() if Config.DATA_SCALING else None

    def generate_response(self, prompt: str, max_length=512, temperature=0.7, reasoning_steps=1):
        for _ in range(reasoning_steps):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            output = self.model.generate(
                **inputs, max_length=max_length, num_beams=5, temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            prompt = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return prompt

    def deep_reasoning(self, prompt, steps=5):
        for i in range(steps):
            temperature = 0.9 - 0.1 * (i / steps)
            prompt = self.generate_response(prompt, temperature=temperature, reasoning_steps=1)
        return prompt

# ========== 训练引擎 ==========
class TrainingEngine:
    def __init__(self, logger: Logger, llm: GapTimeLLM):
        self.logger = logger
        self.llm = llm
        self.data_buffer = deque(maxlen=1000)
        self.thread_pool = ThreadPoolExecutor(max_workers=Config.DISTRIBUTED_WORKERS) if ThreadPoolExecutor else None
        self.load_pretrained_data()

    def load_pretrained_data(self):
        if os.path.exists(Config.PRE_TRAINING_DATA_PATH):
            with open(Config.PRE_TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    prompt, response = line.strip().split("\t", 1)
                    self.data_buffer.append((prompt, response))
            self.logger.log(f"加载{len(self.data_buffer)}条预训练样本", "INFO")

    def add_training_sample(self, prompt: str, response: str):
        if Config.DATA_SCALING:
            prompt = self.scale_text(prompt)
            response = self.scale_text(response)
        self.data_buffer.append((prompt, response))
        self.logger.log(f"新增样本（队列大小:{len(self.data_buffer)}）", "DEBUG")
        self.trigger_training()

    def scale_text(self, text):
        tokens = word_tokenize(text)
        return " ".join([str(len(token)) for token in tokens]) if tokens else text

    def trigger_training(self):
        if (len(self.data_buffer) >= Config.MIN_TRAINING_SAMPLES and
            time.time() - self.llm.last_train_time > Config.AUTO_TRAIN_INTERVAL):
            self.run_distributed_training()

    def run_distributed_training(self):
        if not self.thread_pool:
            return
        self.llm.model.train()
        batch_size = len(self.data_buffer) // Config.DISTRIBUTED_WORKERS
        futures = []
        for i in range(Config.DISTRIBUTED_WORKERS):
            start = i * batch_size
            batch = list(self.data_buffer)[start:start+batch_size]
            futures.append(self.thread_pool.submit(self.train_worker, batch))
        for future in futures:
            future.result()
        self.llm.model.eval()
        self.llm.last_train_time = time.time()
        self.logger.log("分布式训练完成", "INFO")

    def train_worker(self, batch):
        for epoch in range(Config.TRAINING_EPOCHS):
            for prompt, label in batch:
                inputs = self.llm.tokenizer(
                    prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                ).to(self.llm.device)
                labels = self.llm.tokenizer(
                    label, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                )["input_ids"].to(self.llm.device)
                outputs = self.llm.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.llm.optimizer.step()
                self.llm.optimizer.zero_grad()

# ========== 多模态执行器 ==========
class MultiModalExecutor:
    def __init__(self, logger: Logger, llm: GapTimeLLM):
        self.logger = logger
        self.llm = llm
        self.speech_processor = SpeechProcessor(logger)
        self.crawler_pool = ThreadPoolExecutor(max_workers=Config.CRAWLER_THREADS) if ThreadPoolExecutor else None

    def execute_task(self, task: str):
        if not SecurityScanner.scan(task):
            return "危险命令已拦截！"
        if task.startswith("speech:"):
            return self.speech_processor.process_speech_command(task[7:])
        elif task.startswith("plot:"):
            return self.handle_visualization(task[5:])
        elif task.startswith("crawl:"):
            return self.handle_distributed_crawl(task[6:])
        elif task.startswith("image:"):
            return self.handle_image_recognition(task[6:])
        elif task.startswith("install:"):
            return self.install_library(task[8:])
        elif task.startswith("mode:"):
            return self.handle_mode_command(task[5:])
        else:
            return self.llm.deep_reasoning(task, steps=3)

    def handle_visualization(self, data_desc):
        try:
            x = np.linspace(0, 2*np.pi, 100)
            y = np.sin(x) if "正弦" in data_desc else np.cos(x) if "余弦" in data_desc else np.tan(x)
            plt.plot(x, y)
            plt.title(f"Visualization: {data_desc}")
            filename = os.path.join(Config.VISUALIZATION_DIR, f"{data_desc.replace(' ', '_')}.png")
            plt.savefig(filename)
            plt.close()
            return f"图表已生成：{filename}"
        except Exception as e:
            self.logger.log(f"图表生成错误: {str(e)}", "ERROR")
            return "图表生成失败"

    def handle_distributed_crawl(self, urls):
        if not self.crawler_pool or not requests:
            return "分布式爬虫功能不可用（需安装requests和concurrent.futures）"
        results = []
        for url in urls.split(","):
            future = self.crawler_pool.submit(self.crawl_worker, url.strip())
            results.append(future.result(timeout=Config.COMMAND_TIMEOUT))
        return "\n".join(results)

    def crawl_worker(self, url):
        try:
            response = requests.get(url, timeout=10)
            return f"成功爬取 {url}，内容长度：{len(response.text)}"
        except Exception as e:
            return f"爬取 {url} 失败：{str(e)}"

    def handle_image_recognition(self, path):
        if not cv2:
            return "请先安装opencv-python库"
        try:
            img = cv2.imread(path)
            return f"图像尺寸：{img.shape[0]}x{img.shape[1]}，通道数：{img.shape[2]}"
        except Exception as e:
            return f"图像识别失败：{str(e)}"

    def install_library(self, lib):
        if lib not in Config.ALLOWED_LIBS:
            return "禁止安装未授权库！"
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", lib],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return f"{lib} 安装成功"
        except Exception as e:
            return f"{lib} 安装失败：{str(e)}"

    def handle_mode_command(self, mode):
        if mode == "autonomous":
            autonomous_core.toggle_autonomous_mode()
            return "自主模式已切换"
        elif mode == "speech":
            Config.ENABLE_SPEECH_OUTPUT = not Config.ENABLE_SPEECH_OUTPUT
            return f"语音输出已 {'开启' if Config.ENABLE_SPEECH_OUTPUT else '关闭'}"

# ========== 语音处理模块 ==========
class SpeechProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.audio_buffer = deque(maxlen=5)

    def recognize_speech(self):
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                self.audio_buffer.append(audio)
                return self.recognizer.recognize_google(audio, language=Config.SPEECH_RECOGNITION_LANGUAGE)
            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                self.logger.log(f"语音识别错误: {str(e)}", "ERROR")
                return ""

    def speak(self, text):
        if Config.ENABLE_SPEECH_OUTPUT:
            self.engine.say(text)
            self.engine.runAndWait()

    def process_speech_command(self, cmd):
        if cmd == "listen":
            text = self.recognize_speech()
            result = f"识别结果：{text}" if text else "未识别到语音"
            self.speak(result)
            return result
        return "未知语音命令"

# ========== 自主决策核心 ==========
class AutonomousCore:
    def __init__(self, logger: Logger, training_engine: TrainingEngine):
        self.logger = logger
        self.training_engine = training_engine
        self.is_autonomous = False
        self.restart_counter = 0
        self.model_saver = threading.Thread(target=self.auto_save_model, daemon=True)
        self.model_saver.start()

    def auto_save_model(self):
        while True:
            try:
                time.sleep(Config.MODEL_SAVE_INTERVAL)
                if self.is_autonomous:
                    self.training_engine.llm.model.save_pretrained(os.path.join(Config.TEMP_DIR, "autosave_model"))
                    self.logger.log("模型已自动保存", "DEBUG")
            except Exception as e:
                self.logger.log(f"模型保存失败: {str(e)}", "ERROR")

    def toggle_autonomous_mode(self):
        self.is_autonomous = not self.is_autonomous
        self.logger.log(f"自主模式 {'开启' if self.is_autonomous else '关闭'}", "INFO")
        if self.is_autonomous:
            self.start_training_loop()

    def ' start_training_loop(self):
        while self.is_autonomous:
            try:
                self.training_engine.trigger_training()
                time.sleep(Config.AUTO_TRAIN_INTERVAL)
            except Exception as e:
                self.logger.log(f"自主训练错误: {str(e)}", "ERROR")
                self.recover_from_error()

    def recover_from_error(self):
        self.restart_counter += 1
        if self.restart_counter > Config.MAX_RESTARTS:
            self.logger.log("达到最大重启次数，程序终止", "CRITICAL")
            sys.exit(1)
        self.logger.log(f"正在重启程序（剩余次数: {Config.MAX_RESTARTS - self.restart_counter}）", "WARNING")
        subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0])])
        self.shutdown()

    def shutdown(self):
        self.is_autonomous = False
        self.model_saver.join()
        self.logger.log("自主模式关闭", "INFO")

# ========== 主控控制器 ==========
class GapTimeController:
    def __init__(self):
        self.logger = Logger()
        self.llm = GapTimeLLM(self.logger)
        self.training_engine = TrainingEngine(self.logger, self.llm)
        self.executor = MultiModalExecutor(self.logger, self.llm)
        global autonomous_core
        autonomous_core = AutonomousCore(self.logger, self.training_engine)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self.command_listener, daemon=True).start()

    def command_listener(self):
        print(f"\n{Config.APP_NAME} 智能助手 (输入 'help' 查看帮助)")
        while self.running:
            try:
                cmd = input("> 输入命令: ").strip()
                if cmd.lower() == "help":
                    self.show_help()
                elif cmd.lower() == "exit":
                    self.shutdown()
                    break
                else:
                    response = self.executor.execute_task(cmd)
                    self.process_response(cmd, response)
            except Exception as e:
                self.logger.log(f"主循环错误: {str(e)}", "ERROR")
                autonomous_core.recover_from_error()

    def process_response(self, cmd, response):
        self.logger.log(f"用户命令: {cmd} → 响应: {response[:50]}...", "INFO")
        print(f"AI响应: {response}")
        self.executor.speech_processor.speak(response)
        self.training_engine.add_training_sample(cmd, response)

    def show_help(self):
        help_text = """
        可用命令：
        - help                显示帮助信息
        - exit                退出程序
        - speech:listen       语音输入
        - plot:<类型>         生成图表（正弦/余弦）
        - crawl:<网址列表>    分布式爬取网页（逗号分隔）
        - image:<路径>        图像识别
        - install:<库名>      安装指定库（仅限白名单）
        - mode:autonomous     切换自主模式
        - mode:speech         切换语音输出
        """
        print(help_text)
        self.executor.speech_processor.speak("可用命令包括帮助、退出、语音输入、图表生成、分布式爬取、图像识别、库安装和模式切换")

    def shutdown(self):
        self.running = False
        autonomous_core.shutdown()
        self.llm.model.save_pretrained(os.path.join(Config.TEMP_DIR, "shutdown_model"))
        self.logger.log("程序安全关闭", "INFO")
        sys.exit(0)

# ========== 程序入口 ==========
if __name__ == "__main__":
    controller = GapTimeController()
    controller.logger.log(f"{Config.APP_NAME} 启动 (设备: {controller.llm.device})", "INFO")
    controller.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.shutdown()
