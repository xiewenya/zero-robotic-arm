# 训练指南

## 功能特性

✅ **自动保存**
- 最佳模型自动保存到 `./logs/best_model/`
- 每 100,000 步自动保存检查点到 `./models/checkpoints/`
- 按 `Ctrl+C` 随时中断并保存到 `./models/interrupted/`

✅ **恢复训练**
- 支持从任何检查点继续训练
- 自动恢复训练进度（步数、经验回放等）
- 自动加载环境归一化参数

## 使用方法

### 1. 开始新的训练

```bash
python train_robot_arm.py
```

这将：
- 创建新模型并开始训练
- 总训练步数：5,000,000 步
- 自动定期保存检查点

### 2. 从中断处恢复训练

如果您按了 `Ctrl+C` 中断训练：

```bash
python train_robot_arm.py --resume --checkpoint-path ./models/interrupted/td3_robot_arm_interrupted
```

### 3. 从自动检查点恢复训练

从最新的自动检查点恢复：

```bash
python train_robot_arm.py --resume --checkpoint-path ./models/checkpoints/latest_checkpoint
```

从特定步数的检查点恢复：

```bash
python train_robot_arm.py --resume --checkpoint-path ./models/checkpoints/checkpoint_100000
```

### 4. 测试训练好的模型

测试最佳模型：

```bash
python train_robot_arm.py --test --model-path ./logs/best_model/best_model
```

测试最终模型：

```bash
python train_robot_arm.py --test --model-path ./models/td3_robot_arm_final
```

测试指定回合数：

```bash
python train_robot_arm.py --test --model-path ./logs/best_model/best_model --episodes 20
```

## 目录结构

训练过程中会创建以下目录：

```
5. Deep_LR/
├── logs/
│   ├── best_model/              # 最佳模型
│   │   ├── best_model.zip       # 模型文件
│   │   └── vec_normalize.pkl    # 归一化参数
│   └── evaluations.npz          # 评估记录
├── models/
│   ├── checkpoints/             # 定期检查点
│   │   ├── latest_checkpoint.zip
│   │   ├── checkpoint_100000.zip
│   │   ├── checkpoint_200000.zip
│   │   └── vec_normalize.pkl
│   ├── interrupted/             # 中断时的保存
│   │   ├── td3_robot_arm_interrupted.zip
│   │   ├── vec_normalize.pkl
│   │   └── training_info.txt
│   ├── td3_robot_arm_final.zip  # 最终模型
│   └── vec_normalize.pkl
```

## 训练参数

当前配置：
- 总步数：5,000,000
- 评估频率：每 5,000 步
- 检查点保存频率：每 100,000 步
- 学习率：3e-4
- 批次大小：256
- 经验回放缓冲区：3,000,000

## 常见问题

### Q: 训练被意外中断了怎么办？
A: 只要按 `Ctrl+C` 中断过或者运行了至少 100,000 步，就会有检查点保存。使用 `--resume` 参数从最近的检查点恢复。

### Q: 如何查看训练进度？
A: 训练日志会实时显示在终端。同时可以查看 `./models/interrupted/training_info.txt` 了解最后一次保存的步数。

### Q: 恢复训练会保留之前的学习经验吗？
A: 是的！模型的所有参数（网络权重、经验回放缓冲区、优化器状态等）都会被保存和恢复。

### Q: 可以从最佳模型继续训练吗？
A: 可以：
```bash
python train_robot_arm.py --resume --checkpoint-path ./logs/best_model/best_model
```

### Q: 训练需要多长时间？
A: 取决于硬件。使用 GPU 的话，5,000,000 步大约需要几个小时到一天。可以随时中断，稍后继续。

## 提示

1. **定期检查最佳模型**：可以随时测试 `./logs/best_model/best_model` 查看当前最佳效果
2. **多次训练**：可以从最佳模型继续训练，进一步提升性能
3. **保存重要检查点**：如果某个检查点效果不错，可以手动复制保存
4. **GPU 加速**：确保安装了 CUDA 版本的 PyTorch 以使用 GPU 加速

## 示例工作流程

```bash
# 第一天：开始训练
python train_robot_arm.py

# 训练了一段时间后按 Ctrl+C 中断...

# 第二天：继续训练
python train_robot_arm.py --resume --checkpoint-path ./models/interrupted/td3_robot_arm_interrupted

# 训练完成后测试
python train_robot_arm.py --test --model-path ./logs/best_model/best_model --episodes 20
```


