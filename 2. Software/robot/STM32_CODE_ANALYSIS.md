# Zero Robotic Arm STM32 控制代码分析文档

## 一、项目概述

本项目是一个基于 **STM32F407VET6** 微控制器的6轴机械臂控制系统。系统采用 **FreeRTOS** 实时操作系统，通过 **CAN 总线**与闭环步进电机驱动器通信，实现机械臂的精准运动控制。同时支持 **UART 串口命令**控制和可选的 **ESP8266 WiFi/MQTT** 远程控制。

### 1.1 硬件平台
- **MCU**: STM32F407VET6 (168MHz, 512KB Flash, 192KB RAM)
- **电机驱动**: Emm_V5.0 闭环步进电机驱动器 (张大头)
- **通信接口**:
  - CAN1 (PB8/PB9) - 与电机驱动器通信
  - USART1 (PA9/PA10) - 调试/命令输入
  - USART3 (PB10/PB11) - ESP8266 WiFi模块
- **限位开关**: 6路限位开关 (PD0-PD5)，支持上升沿/下降沿中断

### 1.2 软件架构
```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (Application)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ robot.c      │  │robot_cmd.c   │  │robot_kinematics.c│   │
│  │ 核心控制框架  │  │ 命令解析处理  │  │   运动学算法     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     中间件层 (Middleware)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ FreeRTOS     │  │ Emm_V5.c     │  │  esp8266_mqtt.c  │   │
│  │ 任务调度     │  │ 电机驱动协议  │  │   WiFi/MQTT通信  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     驱动层 (HAL Driver)                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │ CAN  │ │ UART │ │ GPIO │ │ DMA  │ │ RNG  │ │ TIM  │     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心功能模块详解

### 2.1 机器人核心控制框架 (robot.c/robot.h)

#### 2.1.1 数据结构设计

**关节数据结构** (`struct joint`):
```c
struct joint {
    float current_angle;              // 当前角度
    enum motor_dir postive_direction; // 关节正方向对应的电机旋转方向
    float reduction_ratio;            // 减速比
    GPIO_TypeDef *limit_gpio_port;    // 限位开关GPIO端口
    uint16_t limit_gpio_pin;          // 限位开关GPIO引脚
    float min_angle;                  // 关节最小角度
    float max_angle;                  // 关节最大角度
    enum dir reset_dir;               // 复位方向
    volatile uint32_t status;         // 关节状态标志位
    float velocity;                   // 关节速度
    float acceleration;               // 关节加速度
};
```

**机器人数据结构** (`struct robot`):
```c
struct robot {
    osThreadId_t control_handle;      // 控制任务句柄
    osThreadId_t cmd_service_handle;  // 命令服务任务句柄
    osThreadId_t mqtt_sync_task_handle; // MQTT同步任务句柄
    float T[4][4];                    // 末端位姿变换矩阵
    struct joint joints[6];           // 6个关节
    uint32_t status;                  // 机器人状态
    QueueHandle_t event_queue;        // 事件队列
    QueueHandle_t cmd_queue;          // 命令队列
    struct position cur_pos;          // 当前末端位置
    struct rotate cur_rot;            // 当前末端姿态
};
```

#### 2.1.2 事件驱动架构

系统采用**事件驱动**的设计模式，支持以下事件类型:

| 事件类型 | 说明 |
|---------|------|
| `ROBOT_JOINT_REL_ROTATE` | 相对旋转：指定关节相对当前位置旋转一定角度 |
| `ROBOT_JOINT_ABS_ROTATE` | 绝对旋转：指定关节旋转到指定角度 |
| `ROBOT_LIMIT_SWITCH_EVENT` | 限位开关触发事件 |
| `ROBOT_AUTO_EVENT` | 自动运动：指定末端位置，算法自动计算各关节角度 |
| `ROBOT_TIMIE_FUNC_EVENT` | 时间函数事件：末端按时间函数P(t)运动 |
| `ROBOT_HARD_RESET_EVENT` | 硬件复位：使用限位开关将所有关节复位到初始位置 |
| `ROBOT_SOFT_RESET_EVENT` | 软件复位：基于编码器反馈将所有关节复位 |
| `ROBOT_JOINT_HARD_RESET_EVENT` | 单关节硬件复位：使用限位开关将指定关节复位 |
| `ROBOT_JOINT_SOFT_RESET_EVENT` | 单关节软件复位：基于编码器反馈将指定关节复位 |
| `ROBOT_READ_STATUS_EVENT` | 读取电机状态：读取指定关节的所有状态参数 |
| `ROBOT_REMOTE_CONTROL_EVENT` | 远程控制：根据手柄指令实时控制机械臂 |
| `ROBOT_JOINTS_SYNC_EVENT` | 关节同步：同步机械臂关节状态 |

#### 2.1.3 多任务架构

系统运行以下FreeRTOS任务:

1. **robot_control_task** (优先级: osPriorityRealtime3, 栈大小: 4096字节)
   - 核心控制任务，从事件队列接收并处理机械臂运动事件
   - 负责关节旋转、自动运动、复位等核心功能

2. **robot_cmd_service** (优先级: osPriorityRealtime2, 栈大小: 2048字节)
   - 命令服务任务，处理UART和MQTT接收的命令
   - 解析命令并转换为对应的事件

3. **defaultTask** (优先级: osPriorityNormal, 栈大小: 512字节)
   - 默认任务，负责初始化机器人系统

---

### 2.2 运动学算法 (robot_kinematics.c)

#### 2.2.1 DH参数模型

系统采用标准的**D-H (Denavit-Hartenberg)** 参数模型描述6轴机械臂的运动学关系:

```c
const float D_H[6][4] = {
    {0,      0,        0,        M_PI/2},   // 关节1
    {0,      M_PI/2,   0,        M_PI/2},   // 关节2
    {200,    M_PI,     0,        -M_PI/2},  // 关节3 (连杆长度200mm)
    {47.63,  -M_PI/2,  -184.5,   0},        // 关节4
    {0,      M_PI/2,   0,        M_PI/2},   // 关节5
    {0,      M_PI/2,   0,        0}         // 关节6
};
```

DH参数说明: `[a_i, α_i, d_i, θ_i]`
- `a_i`: 连杆长度 (mm)
- `α_i`: 连杆扭转角 (rad)
- `d_i`: 连杆偏距 (mm)
- `θ_i`: 关节角偏移 (rad)

#### 2.2.2 逆运动学求解

逆运动学算法流程:

```
输入: 目标末端位姿矩阵 T_target (4x4)
       ↓
┌─────────────────────────────┐
│  1. 计算 θ3 (关节3角度)      │  基于几何关系求解
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│  2. 计算 θ2 (关节2角度)      │  依赖θ3
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│  3. 计算 θ1 (关节1角度)      │  依赖θ2, θ3
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│  4. 计算 θ5 (关节5角度)      │  基于旋转矩阵ZYZ分解
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│  5. 计算 θ4 (关节4角度)      │  依赖θ1-θ3, θ5
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│  6. 计算 θ6 (关节6角度)      │  依赖θ1-θ5
└─────────────────────────────┘
       ↓
输出: 4组可能的关节角度解
```

**最优解选择策略**: 
- 计算每组解与当前关节位置的加权差值
- 各关节权重: `[5, 3, 3, 1, 1, 1]` (基座关节权重最高)
- 选择加权差值最小的解作为最优解
- 自动过滤超出关节限位范围的无效解

#### 2.2.3 路径插值

**线性插值** (`robot_path_interpolation_linear`):
- 计算起点到终点的直线距离
- 按照 `ROBOT_INTERPOLATION_RESOLUTION` (1.0mm) 的分辨率生成路径点
- 确保末端执行器沿直线轨迹运动

**时间函数插值** (`robot_time_func_path_interpolation`):
- 根据时间函数计算各时刻的末端位置
- 按照 `ROBOT_INTERPOLATION_TIME_RESOLUTION` (100ms) 的时间分辨率采样
- 内置示例: 圆形轨迹函数 `time_func_circle`

---

### 2.3 PID运动控制

系统采用**速度PID控制**实现平滑的关节运动:

```c
// PID参数
#define ROBOT_PID_KP  10.0f    // 比例系数
#define ROBOT_PID_KI  0.002f   // 积分系数
#define ROBOT_PID_KD  0.0f     // 微分系数
#define ROBOT_PID_PERIOD 20    // PID周期 (ms)
```

**PID控制流程** (`robot_pid_one_period`):
```
1. 读取当前关节角度 (通过CAN从编码器获取)
2. 计算角度误差 = 目标角度 - 当前角度
3. 累积积分误差
4. 计算PID输出: v = Kp*error + Ki*intg_error + Kd*d_error
5. 发送速度指令到电机驱动器
6. 等待下一个PID周期
```

**角度差值处理** (`robot_angle_diff`):
- 自动处理0°/360°跨越问题
- 确保计算结果在-180°到180°范围内

---

### 2.4 电机驱动接口 (Emm_V5.c)

系统使用**张大头Emm_V5.0闭环步进驱动器**，通过CAN总线通信。

#### 2.4.1 主要控制函数

| 函数 | 功能 |
|------|------|
| `Emm_V5_Vel_Control` | 速度模式控制 (0-5000 RPM) |
| `Emm_V5_Pos_Control` | 位置模式控制 (脉冲计数) |
| `Emm_V5_Stop_Now` | 立即停止电机 |
| `Emm_V5_Read_Sys_Params` | 读取系统参数 (位置、速度、状态等) |
| `Emm_V5_Reset_CurPos_To_Zero` | 将当前位置清零 |
| `Emm_V5_En_Control` | 使能/禁用电机 |
| `Emm_V5_Origin_Trigger_Return` | 触发回零动作 |

#### 2.4.2 CAN通信协议

**发送帧格式**:
- 扩展帧ID: `(电机地址 << 8) | 包序号`
- 数据长度: 最大8字节
- 功能码在数据域第1字节

**接收处理** (`HAL_CAN_RxFifo0MsgPendingCallback`):
- 解析电机地址: `(ExtId >> 8) - 1`
- 检测电机到位信号: `0xFD 0x9F 0x6B`
- 设置关节`ROBOT_STATUS_READY`状态位

---

### 2.5 命令处理系统 (robot_cmd.c)

#### 2.5.1 UART命令格式

命令格式: `<命令名> <参数1> <参数2> ... <参数6>\n`

| 命令 | 参数 | 说明 |
|------|------|------|
| `remote_enable` | 无 | 启用远程手柄控制模式 |
| `remote_disable` | 无 | 禁用远程控制模式 |
| `remote_event` | vx vy rx ry lt rt | 远程控制速度指令 |
| `rel_rotate` | joint_id angle | 相对旋转指定关节（带角度范围检查） |
| `abs_rotate` | joint_id angle | 绝对旋转到指定角度（带角度范围检查） |
| `auto` | x y z | 自动运动到目标位置 |
| `hard_reset` | 无 | 硬件复位所有关节 (使用限位开关) |
| `soft_reset` | 无 | 软件复位所有关节 (基于编码器) |
| `joint_hard_reset` | joint_id | 硬件复位单个关节 (使用限位开关) |
| `joint_soft_reset` | joint_id | 软件复位单个关节 (基于编码器) |
| `read_angle` | joint_id | 读取指定关节的当前角度 |
| `read_all_angles` | 无 | 读取所有关节的当前角度 |
| `read_status` | joint_id | 读取指定关节的所有状态参数 |
| `zero` | 无 | 将所有关节编码器清零 |

#### 2.5.2 MQTT命令格式

MQTT消息格式: `[MCU][TYPE][ARG0 ARG1 ARG2 ARG3 ARG4 ARG5]`

- `TYPE`: 事件类型编号
- `ARG0-5`: 6个浮点参数

---

### 2.6 ESP8266 WiFi/MQTT通信 (esp8266_mqtt.c)

#### 2.6.1 初始化流程

```
1. 发送 AT+RST 重启ESP8266
2. 设置STA模式 AT+CWMODE=1
3. 连接WiFi AT+CWJAP="SSID","PASSWORD"
4. 配置MQTT用户属性 AT+MQTTUSERCFG
5. 连接MQTT服务器 AT+MQTTCONN
6. 订阅主题 AT+MQTTSUB
7. 启用UART3接收中断
```

#### 2.6.2 消息发布

使用 `esp8266_publish_message` 函数发布MQTT消息:
```c
int esp8266_publish_message(const char *topic, const char *message, 
                           uint8_t qos, uint32_t wait);
```

**关节状态同步消息格式**:
```
[PC][ROBOT_JOINTS_SYNC_EVENT][θ1 θ2 θ3 θ4 θ5 θ6]
```

---

### 2.7 限位开关与复位系统

#### 2.7.1 限位开关硬件接线

**接线方式**: NC (常闭) → GPIO, C (公共端) → GND

**电气逻辑**:
- 未触发（NC导通）: GPIO = LOW (GND)
- 触发时（NC断开）: GPIO = HIGH (内部上拉)

**GPIO配置**:
- 模式: `GPIO_MODE_IT_RISING_FALLING` (上升沿和下降沿中断)
- 上拉: `GPIO_PULLUP` (启用内部上拉电阻)
- 端口: GPIOD (PD0-PD5)

#### 2.7.2 限位开关中断处理

```c
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    // 1. 根据GPIO引脚识别关节ID
    // 2. 检查限位开关是否使能
    // 3. 防抖处理 (检查LIMIT_HAPPENED状态位)
    // 4. 立即停止电机
    // 5. 发送限位开关事件到队列
}
```

**安全保护机制**:
- ✅ 限位开关在系统初始化时**全部启用**
- ✅ 在所有电机运动过程中**始终保持启用状态**
- ✅ 触发后立即从中断中停止电机，确保快速响应
- ✅ 防抖处理避免重复触发

#### 2.7.3 硬件复位流程 (`robot_joint_hard_reset`)

#### 2.7.4 硬件复位流程 (`robot_joint_hard_reset`)

```
对于每个关节 (从关节5到关节0):
  1. 读取当前限位状态
  2. 如果已触发：清零编码器，清除限位状态标志，返回
  3. 如果未触发：向复位方向旋转
  4. 等待限位开关触发
  5. 停止电机并清零编码器
  6. 立即清除限位状态标志（不依赖异步事件处理）
  7. 恢复关节初始角度
```

**重要改进**：
- 限位开关在系统启动时已全局启用，运行期间始终保持启用
- 复位完成后**立即清除**限位状态标志，确保后续运动不受影响
- 不依赖异步事件处理，避免事件队列阻塞导致的状态延迟

#### 2.7.5 软件复位流程 (`robot_joint_soft_reset`)

```
对于每个关节 (从关节5到关节0):
  1. 从编码器读取当前角度
  2. 映射到有效角度范围
  3. 计算到初始位置的最短路径方向
  4. 旋转到初始角度
  5. 更新当前位置
```

---

### 2.8 远程手柄控制 (robot_joystick.py)

#### 2.8.1 功能说明

Python脚本用于通过游戏手柄远程控制机械臂，主要功能:

- 读取游戏手柄各轴的模拟值
- 将手柄数据转换为机械臂速度指令
- 通过串口发送给STM32

#### 2.8.2 控制映射

```
左摇杆X轴 → 末端Y方向速度 (vy)
左摇杆Y轴 → 末端X方向速度 (vx)
右摇杆X轴 → 关节5旋转速度 (ry)
右摇杆Y轴 → 关节4旋转速度 (rx)
LT/RT扳机 → 末端Z方向速度 (vz = (LT-RT)/2)
```

最大线速度: 20 mm/s
最大角速度: 5 RPM

---

## 三、系统初始化流程

```
main()
  ├── HAL_Init()                    // HAL库初始化
  ├── SystemClock_Config()          // 配置168MHz主频
  ├── MX_GPIO_Init()               // GPIO初始化 (LED, 限位开关)
  ├── MX_DMA_Init()                // DMA初始化
  ├── MX_CAN1_Init()               // CAN1初始化 (500Kbps)
  ├── MX_RNG_Init()                // 随机数发生器初始化
  ├── MX_USART1_UART_Init()        // USART1初始化 (115200bps)
  ├── MX_USART3_UART_Init()        // USART3初始化 (115200bps)
  ├── USER_CAN1_Filter_Init()      // CAN过滤器配置
  ├── HAL_CAN_Start()              // 启动CAN控制器
  ├── HAL_CAN_ActivateNotification()  // 使能CAN接收中断
  ├── osKernelInitialize()         // 初始化FreeRTOS内核
  ├── MX_FREERTOS_Init()           // 创建默认任务
  │     └── StartDefaultTask()
  │           └── robot_init()     // 机器人系统初始化
  │                 ├── 初始化关节参数
  │                 ├── 创建事件队列
  │                 ├── 创建命令队列
  │                 ├── 创建robot_control_task
  │                 └── 创建robot_cmd_service任务
  └── osKernelStart()              // 启动任务调度器
```

---

## 四、关键参数配置

### 4.1 关节配置表

| 关节 | 初始角度 | 正方向 | 减速比 | 角度范围 | 复位方向 |
|------|---------|--------|--------|----------|---------|
| 1 | 90° | CCW | 50:1 | 0°~360° | 负方向 |
| 2 | 90° | CW | 50.89:1 | 90°~180° | 负方向 |
| 3 | -90° | CW | 50.89:1 | -90°~90° | 负方向 |
| 4 | 0° | CW | 51:1 | -90°~90° | 负方向 |
| 5 | 90° | CCW | 26.85:1 | 0°~90° | 正方向 |
| 6 | 0° | CW | 51:1 | 0°~360° | 负方向 |

### 4.2 运动参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 默认关节速度 | 10 RPM | 正常运动速度 |
| 默认加速度 | 200 | 加速度等级 |
| 复位速度 | 10 RPM | 复位时的速度 |
| 复位加速度 | 100 | 复位时的加速度 |
| 路径插值分辨率 | 1.0 mm | 直线轨迹的点间距 |
| 时间插值分辨率 | 100 ms | 时间函数采样周期 |
| PID控制周期 | 20 ms | PID更新频率 |
| CAN通信超时 | 10 ms | CAN收发超时时间 |

### 4.3 状态标志位

| 标志位 | 说明 |
|--------|------|
| `ROBOT_STATUS_LIMIT_ENABLE` | 限位开关使能（系统启动时自动设置，运行期间始终保持） |
| `ROBOT_STATUS_LIMIT_HAPPENED` | 限位开关已触发（用于防抖和事件处理） |
| `ROBOT_STATUS_READY` | 当前运动已完成 |
| `ROBOT_STATUS_RMODE_ENABLE` | 远程控制模式使能 |
| `ROBOT_STATUS_MQTT_CONNECTED` | MQTT连接状态 |

---

## 五、文件结构说明

```
robot/
├── Core/
│   ├── Inc/                       # 头文件
│   │   ├── robot.h               # 机器人核心定义
│   │   ├── robot_cmd.h           # 命令处理接口
│   │   ├── robot_kinematics.h    # 运动学接口
│   │   ├── Emm_V5.h              # 电机驱动接口
│   │   ├── esp8266_mqtt.h        # WiFi/MQTT接口
│   │   ├── can.h                 # CAN通信接口
│   │   ├── usart.h               # 串口接口
│   │   └── ...                   # 其他HAL配置头文件
│   │
│   ├── Src/                       # 源文件
│   │   ├── main.c                # 主程序入口
│   │   ├── robot.c               # 机器人核心控制 (1106行)
│   │   ├── robot_cmd.c           # 命令解析处理 (179行)
│   │   ├── robot_kinematics.c    # 运动学算法 (499行)
│   │   ├── Emm_V5.c              # 电机驱动协议 (341行)
│   │   ├── esp8266_mqtt.c        # WiFi/MQTT通信 (150行)
│   │   ├── can.c                 # CAN通信实现 (217行)
│   │   ├── usart.c               # 串口通信实现 (288行)
│   │   ├── freertos.c            # FreeRTOS配置
│   │   ├── gpio.c                # GPIO配置
│   │   └── ...                   # 其他HAL/系统文件
│   │
│   └── Startup/                   # 启动文件
│
├── Drivers/                       # STM32 HAL驱动库
├── Middlewares/                   # FreeRTOS中间件
├── robot.ioc                      # STM32CubeMX工程文件
├── robot_joystick.py             # 手柄控制Python脚本
└── STM32F407VETX_FLASH.ld        # 链接脚本
```

---

## 六、使用示例

### 6.1 串口命令控制

```bash
# 软件复位所有关节到初始位置
soft_reset

# 硬件复位所有关节（使用限位开关）
hard_reset

# 单关节软件复位（关节0）
joint_soft_reset 0

# 单关节硬件复位（关节1）
joint_hard_reset 1

# 读取关节0的当前角度
read_angle 0

# 读取所有关节的当前角度
read_all_angles

# 读取关节0的所有状态参数（位置、速度、电流、电压、标志位等）
read_status 0

# 关节1相对旋转30度
rel_rotate 0 30

# 自动运动到位置(10, 20, 30)mm
auto 10 20 30

# 启用手柄远程控制
remote_enable

# 禁用手柄远程控制
remote_disable
```

### 6.2 手柄控制

```bash
# 运行手柄控制脚本
python robot_joystick.py -p COM3 -b 115200
# macOS/Linux
python robot_joystick.py -p /dev/ttyUSB0 -b 115200
```

---

## 七、扩展与修改指南

### 7.1 添加新命令

1. 在 `robot_cmd.c` 中添加命令处理函数
2. 在 `robot_uart1_cmd_table` 数组中注册命令
3. 如需新事件类型，在 `robot.h` 的 `robot_event_type` 枚举中添加

### 7.2 修改DH参数

修改 `robot.c` 中的 `D_H` 数组和 `g_joints_init` 数组以适配不同机械臂结构。

### 7.3 修改PID参数

调整 `robot.h` 中的 `ROBOT_PID_KP`, `ROBOT_PID_KI`, `ROBOT_PID_KD` 宏定义。

### 7.4 添加新的时间函数轨迹

1. 在 `robot.c` 中实现新的时间函数 (参考 `time_func_circle`)
2. 将 `g_robot_time_func` 指向新函数

---

## 八、总结

本STM32机械臂控制代码实现了一个完整的6轴机械臂控制系统，主要特点包括:

1. **模块化设计**: 核心控制、运动学算法、电机驱动、通信接口各自独立
2. **实时性保证**: 基于FreeRTOS的多任务架构，确保控制实时性
3. **多种控制方式**: 支持单关节控制、笛卡尔空间控制、轨迹跟踪、手柄遥控
4. **完善的安全机制**: 限位开关保护、硬件/软件复位功能
5. **可扩展的通信**: 串口调试、WiFi/MQTT远程控制
6. **精确的运动学**: 基于DH参数的逆运动学求解，支持多解优选

代码结构清晰，注释完善，便于二次开发和功能扩展。
