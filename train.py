import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from ultralytics import YOLO

# ==================== 配置参数 ====================
# 模型配置文件路径
MODEL_CONFIG_PATH = "ultralytics/cfg/models/11/yolo11.yaml"

# 训练参数
EPOCHS = 100          # 训练轮次，越高越好（但需要更多时间）
BATCH_SIZE = 64       # 批次大小，越高越好（但受显存限制）
LEARNING_RATE = 0.01  # 学习率，适中最好（太高会震荡，太低收敛慢）
MOMENTUM = 0.937      # 动量，越高越好（0.9-0.99之间）
WEIGHT_DECAY = 0.0005 # 权重衰减，适中最好（防止过拟合）
WARMUP_EPOCHS = 5     # 预热轮次，适中最好（帮助稳定训练）
PATIENCE = 50         # 耐心值，越高越好（早停等待轮次）
IMAGE_SIZE = 320      # 图像尺寸，越高越好（但计算量更大）
MIXUP = 0.15          # Mixup增强，适中最好（0.0-0.3之间）
COPY_PASTE = 0.3      # Copy-paste增强，适中最好（0.0-0.5之间）

# 数据集和设备
DATA_CONFIG = "datasets/gwhd_2021_yolo/data.yaml"
DEVICE = '0,1,2,3'

# ==================== 训练 ====================
if __name__ == "__main__":
    # 创建模型
    model = YOLO(MODEL_CONFIG_PATH)
    # model.load("yolo11n.pt")
    
    # 训练模型
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        lr0=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        project="runs/detect",
        name=f"train_{MODEL_CONFIG_PATH.split('/')[-1].split('.')[0]}",
        save_period=10,
        val=True,
        plots=True,
        verbose=True
    )
    
    # 验证模型
    val_results = model.val()
    
    print(f"训练完成!")
    print(f"最佳mAP50: {val_results.box.map50:.3f}")
    print(f"最佳mAP50-95: {val_results.box.map:.3f}")