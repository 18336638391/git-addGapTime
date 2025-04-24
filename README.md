
导入操作系统
导入系统
导入时间
导入线程
导入回溯
导入子流程
导入日期时间
从集合导入退出
将numpy导入为np
从键入导入Dict，Any
进口火炬
从变压器导入GPT2LMHeadModel、GPT2Tokenizer、管道

#========== 全局配置 ==========
类配置：
app_NAME="GapTimeLLM"
log_DIR="./logs"
temp_DIR="./temp"
Max_RESTARTS=5
restart_DELAY=10
event_HISTORY_SIZE=100
LLM_MODEL="gpt2-medium"#可替换为本地模型路径
time_DIM=128
feedback_LOG=os.path.join(LOG_DIR，"user_feedback.log")
upgrade_TRIGGER_INTERVAL=3600#每小时检查升级
code_TEMPLATE_PATH="gap_time_template.py"
command_TIMEOUT=20
# 新增训练相关配置
training_DATA_PATH=os.path.join(TEMP_DIR，"training_data.csv")
learning_RATE=5e-5
training_epochs=10
min_TRAINING_SAMPLES=50#最小训练样本数
auto_TRAIN_INTERVAL=3600#自动训练间隔（秒）

#========== 大语言模型核心 ==========
类GapTimeLLM：
定义__init__(自身，记录器)：
self.logger=记录器
self.device="cuda"(如果是火炬)。cuda.is_available()else"cpu"
self.model=GPT2LMHeadModel.From_pretrained(配置LLM_MODEL).To(self.设备)
self.tokenizer=GPT2Tokenizer.From_pretrained(配置LLM_MODEL)
self.model.eval()#初始推理模式
self.last_train_time=0#记录最后训练时间

    Def generate_response(self，提示符：str，max_length=512)：
    """核心文本生成"""
    输入=self.tokenizer(提示，return_tensors="pt").to(self.device)
    输出=自模型生成(**输入，max_length=max_length，num_beams=5)
    返回self.tokenizer.decode(输出[0]，skip_special_tokenes=True)

    Def切换至培训模式(自)：
    """切换训练模式"""
    self.model.train()
    self.logger.log("模型进入训练模式"，"信息")

    Def切换至推理模式(自身)：
    """切换推理模式"""
    self.model.eval()
    self.logger.log("模型进入推理模式"，"信息")

    定义保存模型(self，路径：str)(_M)：
    """保存训练后模型"""
    self.model.save_pretrained(路径)
    self.tokenizer.save_pretrained(路径)
        self.logger.log(f"模型保存至 {path}", "INFO")

# ========== 深度训练引擎 ==========
class TrainingEngine:
    def __init__(self, logger: Logger, llm: GapTimeLLM):
        self.logger = logger
        self.llm = llm
        self.optimizer = torch.optim.AdamW(llm.model.parameters(), lr=Config.LEARNING_RATE)
        self.data_buffer = []  # 训练数据缓冲区

    def add_training_sample(self, prompt: str, response: str):
        """添加训练样本"""
        self.data_buffer.append((prompt, response))
        self.logger.log(f"新增训练样本（总数:{len(self.data_buffer)}）", "DEBUG")

    def has_enough_data(self) -> bool:
        """检查是否达到最小训练样本"""
        return len(self.data_buffer) >= Config.MIN_TRAINING_SAMPLES

    def run_training_loop(self):
        """完整训练流程"""
        if not self.has_enough_data():
            self.logger.log(f"训练数据不足（需要{Config.MIN_TRAINING_SAMPLES}，当前{len(self.data_buffer)}）", "WARNING")
            return

        self.llm.switch_to_training_mode()
        self.logger.log(f"开始模型训练（轮次:{Config.TRAINING_EPOCHS}）", "INFO")

        for epoch in range(Config.TRAINING_EPOCHS):
            total_loss = 0.0
            for prompt, label in self.data_buffer:
                inputs = self.llm.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
                labels_inputs = self.llm.tokenizer(label, return_tensors="pt").to(self.llm.device)
                
                outputs = self.llm.model(**inputs, labels=labels_inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger.log(f"Epoch {epoch+1} 损失: {total_loss/len(self.data_buffer):.4f}", "INFO")

        self.llm.switch_to_inference_mode()
        self.llm.last_train_time = time.time()
        self.llm.save_model(os.path.join(Config.TEMP_DIR, "fine_tuned_model"))
        self.data_buffer.clear()  # 清空已训练数据
        self.logger.log("训练完成，模型已保存", "INFO")

# ========== 任务执行系统 ==========
class TaskExecutor:
    def __init__(self, logger: Logger, llm: GapTimeLLM):
        self.logger = logger
        self.llm = llm

    def execute_task(self, task: str):
        """智能任务执行（支持命令/文本生成）"""
        if task.startswith("system:"):  # 系统命令
            cmd = task[7:].strip()
            return self.execute_system_command(cmd)
        else:  # 文本生成任务
            return self.llm.generate_response(task)

    def execute_system_command(self, command: str):
        """安全执行系统命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=Config.COMMAND_TIMEOUT
            )
            if result.returncode == 0:
                return f"命令执行成功:\n{result.stdout}"
            else:
                return f"命令执行失败:\n{result.stderr}"
        except Exception as e:
            return f"命令执行异常: {str(e)}"

# ========== 自主决策核心 ==========
class AutonomousCore:
    def __init__(self, logger: Logger, llm: GapTimeLLM, training_engine: TrainingEngine):
        self.logger = logger
        self.llm = llm
        self.training_engine = training_engine
        self.event_history = deque(maxlen=Config.EVENT_HISTORY_SIZE)
        self.user_commands = deque()

    def record_user_command(self, command: str):
        """记录用户命令用于训练"""
        response = self.llm.generate_response(f"处理命令: {command}")
        self.training_engine.add_training_sample(command, response)
        self.event_history.append(f"命令: {command} → 响应: {response[:50]}...")

    def trigger_auto_train(self):
        """定时自动训练触发"""
        if (time.time() - self.llm.last_train_time) > Config.AUTO_TRAIN_INTERVAL:
            self.logger.log("触发定时自动训练", "INFO")
            self.training_engine.run_training_loop()

    def analyze_event_history(self):
        """深度事件分析（示例：提取高频命令）"""
        command_counts = {}
        for event in self.event_history:
            if "命令:" in event:
                cmd = event.split("命令: ")[1].split(" → ")[0]
                command_counts[cmd] = command_counts.get(cmd, 0) + 1
        if command_counts:
            self.logger.log(f"高频命令: {max(command_counts, key=command_counts.get)}", "DEBUG")

# ========== 主控AI程序核心 ==========
class GapTimeController:
    def __init__(self):
        self.logger = Logger()
        self.llm = GapTimeLLM(self.logger)
        self.training_engine = TrainingEngine(self.logger, self.llm)
        self.task_executor = TaskExecutor(self.logger, self.llm)
        self.autonomous_core = AutonomousCore(self.logger, self.llm, self.training_engine)
        self.running = False
        self.init_environment()

    def init_environment(self):
        """初始化环境"""
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        # 加载历史训练模型（如有）
        model_path = os.path.join(Config.TEMP_DIR, "fine_tuned_model")
        if os.path.exists(model_path):
            self.llm.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.llm.device)
            self.logger.log("加载上次训练的模型", "INFO")

    def start_autonomous_mode(self):
        """启动自主运行模式"""
        if self.running:
            self.logger.log("自主模式已运行", "INFO")
            return
        self.running = True
        threading.Thread(target=self.autonomous_loop, daemon=True).start()
        self.logger.log("自主模式启动", "INFO")

    def autonomous_loop(self):
        """包含自我迭代的主循环"""
        while self.running:
            try:
                self.autonomous_core.analyze_event_history()
                self.autonomous_core.trigger_auto_train()
                time.sleep(60)  # 每分钟执行一次自主逻辑
            except Exception as e:
                self.logger.log(f"自主循环异常: {str(e)}", "ERROR")
                traceback.print_exc()

    def handle_user_command(self, command: str):
        """全功能命令处理"""
        self.autonomous_core.record_user_command(command)
        
        if command.startswith("train now"):
            self.training_engine.run_training_loop()
        elif command.startswith("execute "):
            task = command[8:]
            result = self.task_executor.execute_task(task)
            self.logger.log(f"任务结果: {result[:100]}...", "INFO")
            print(result)
        elif command == "status":
            self.print_status()
        else:
            response = self.llm.generate_response(f"用户命令: {command}")
            print(f"AI响应: {response[:200]}...")

    def print_status(self):
        """显示系统状态"""
        print(f"模型状态: {'训练中' if self.llm.model.training else '推理中'}")
        print(f"训练数据: {len(self.training_engine.data_buffer)}/{Config.MIN_TRAINING_SAMPLES}")
        print(f"最后训练时间: {datetime.datetime.fromtimestamp(self.llm.last_train_time).strftime('%Y-%m-%d %H:%M')}")

    def command_listener(self):
        """交互式命令控制台"""
        print(f"\n{Config.APP_NAME} 命令系统 (输入 'help' 查看帮助)")
        while True:
            try:
                cmd = input("> 输入命令: ").strip()
                if cmd == "help":
                    self.show_help()
                elif cmd == "exit":
                    self.shutdown()
                    break
                else:
                    self.handle_user_command(cmd)
            except EOFError:
                self.shutdown()
                break

    def show_help(self):
        """命令帮助"""
        print("""
        可用命令:
        - train now          立即执行模型训练
        - execute <task>     执行任务（支持 'system:命令' 或 文本生成）
        - status             查看系统状态
        - exit               退出程序
        """)

    def shutdown(self):
        """优雅关闭"""
        self.running = False
        self.llm.save_model(os.path.join(Config.TEMP_DIR, "shutdown_model"))
        self.logger.log("程序优雅关闭，模型已保存", "INFO")
        sys.exit(0)

# ========== 日志系统 ==========
class Logger:
    def __init__(self):
        self.log_file = os.path.join(Config.LOG_DIR, f"{Config.APP_NAME}_{time.strftime('%Y%m%d')}.log")
        os.makedirs(Config.LOG_DIR, exist_ok=True)

    def log(self, message: str, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

# ========== 程序入口 ==========
if __name__ == "__main__":
    gap_time = GapTimeController()
    gap_time.logger.log(
        f"{Config.APP_NAME} 启动 (设备: {gap_time.llm.device})",
        "INFO"
    )
