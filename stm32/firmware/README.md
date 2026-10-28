# STM32 控制板 HAL 模块框架

对应上一级四份控制板文档的五个模块。目标芯片为 STM32F103C8T6，64 KiB Flash、20 KiB RAM，使用已安装的 STM32CubeF1 v1.8.7 / HAL1。

这是后续开发的可编译基础：包含板级初始化和基础状态逻辑；串口完整协议、OLED 控制器驱动仍是明确的接入点。尚未进行开发板运行或外围实物验证。

## 从这里开始

在终端执行：

```bat
D:\IcicleMeasurement\stm32\firmware\build.cmd
```

生成 `build/ice_control.elf`、`build/ice_control.hex` 和 `build/ice_control.map`。脚本只编译，不连接下载器、不烧写。也可用现有 VS Code 打开本目录后在终端运行 `build.cmd`。当前没有 CubeMX 生成的 `.ioc` 或 CubeIDE 工程元数据；不能直接宣称已完成 CubeIDE 图形构建。

默认引用 `D:/STM32Tools` 下已安装的工具和 SDK，不向仓库复制 HAL/CMSIS。工具位置变化时，修改 `build.cmd` 的 CMake/Ninja 路径，并通过 CMake 参数 `STM32_GCC_ROOT`、`STM32CUBE_F1_ROOT` 指定 GCC 和 SDK。更换编译器后使用新的构建目录。

## 文件与文档的对应关系

| 文档中的模块 | 代码 | 当前内容 |
| --- | --- | --- |
| 模块1 USART1 | `Modules/communication.*` | 115200 8N1、中断接收、256字节环形缓冲、错误后丢弃残帧、异步发送接口 |
| 模块1 协议接入 | `Modules/protocol.*` | 接口和诊断计数；编解码未实现，默认丢弃输入、发送返回失败 |
| 模块2 OLED | `Modules/display.*` | 三行状态文本、整数十分之一毫米显示、100ms刷新节拍 |
| 模块2 屏幕接入 | `Modules/oled_port.*` | 未指定控制器或地址，初始化和绘制明确返回未就绪 |
| 模块4 双LED | `Modules/status_led.*` | 在线且无故障为绿灯，其余为红灯 |
| 模块6 两键 | `Modules/buttons.*` | 30ms防抖、每次按下一次事件，无长按或组合键 |
| 模块7 照明 | `Modules/lighting.*` | 0–100%占空比，越界输入关灯；默认0% |
| 模块协作 | `App/app_control.*` | 连接、同步、测量、结果、停止及故障状态逻辑，独立于HAL |
| 芯片和引脚 | `Board/board.*` | HSI、GPIO、USART1、I2C1、TIM3初始化 |
| 程序入口 | `Core/main.c` | 先处理停止/超时，再处理有限数量串口字节，最后更新显示 |
| 中断入口 | `Core/stm32f1xx_it.c` | SysTick、USART1和异常时关灯 |

## 当前默认运行行为

- 上电输出关灯、红灯，状态为未连接；没有自动测量或自动亮灯演示。
- 协议适配器未实现，因此接入USB-TTL发送任意文字不会建立连接，也不会收到ACK。可在调试器中查看 `controller`、`display_frame`、`protocol_stats`、`communication_stats`。
- 显示文本已准备，但 OLED 不会点亮；I²C初始化不等于屏幕驱动完成。
- 原始STOP低电平在主循环中立即禁止照明，不等待串口；防抖后的STOP事件只发送一次停止请求。持续按住STOP期间，任何同步或开灯事件都不能开灯。
- START仅在已连接、已同步且READY时尝试发送；发送成功后等待上位机状态确认，不自行进入MEASURING。
- 3秒无有效心跳清除结果并关灯。恢复心跳只恢复链路标记；必须重新同步，照明还要新的有效命令。

主循环是裸机轮询，无RTOS。STOP是普通操作停止，响应依赖主循环正常执行；不是硬件安全急停。新增驱动应保持单次服务短且有界，避免长延时挡住按键和串口处理。

## 接入串口协议

先统一帧头、长度、命令字段、字节序、单位、校验算法及会话/任务标识，再替换 `protocol.c` 的接入函数。`AppEventKind` 是内部C事件，不是已经确定的线协议编号。

1. `Protocol_Feed` 在主循环中逐字节解析，只对完整、校验通过、属于当前会话和任务的帧调用 `App_HandleHostEvent`。错误帧和半帧不得产生业务事件。
2. 有效心跳发出 `APP_HOST_HEARTBEAT`，应用通过发送回调请求ACK。若发送缓冲忙，协议适配器自行安排有界重试；不可报告成功却丢弃命令。
3. 连接建立、STOP/取消或故障后的恢复，需要新握手得到 `APP_HOST_SYNC`。不能把旧缓存状态帧当成同步完成；普通心跳也不等于同步。
4. 上位机确认开测才发送 `APP_HOST_STATE(APP_MEASURING)`；本轮结果在MEASURING期间提交 `APP_HOST_RESULT`，随后可提交READY。结果使用 `uint32_t` 的0.1mm单位，例如1234代表123.4mm。取消、拒绝和新测量会清除旧结果。
5. 亮度事件值只能为0–100，不能用LIGHT_ON/OFF代替参数。`APP_HOST_FAULT` 关灯并要求重新同步。
6. `Protocol_Send` 编码START/STOP/ACK后调用 `Communication_WriteFrame`。当前传输接口单帧上限64字节，无发送队列；若最终协议需要更长帧，应同步修改容量并测试。返回false时表示没有成功提交发送。
7. `Protocol_Reset` 必须丢弃部分帧及旧会话解析状态。接收溢出、UART错误、STOP和失联会清理接收路径，旧帧不能跨会话继续执行。

不要在USART1协议流中混入任意printf日志；当前工程未添加文件系统、堆分配或半主机调用。

## 接入OLED

确认实际控制器、3.3V供电、I²C上拉和地址后，实现 `oled_port.c`。上层提供LINK、STATE、距离/NO RESULT三行ASCII；不涉及中文字体库。`Board_I2cWrite` 接受7位地址，每次最多32字节、超时5ms。驱动应分批或异步更新，不能在一次 `OledPort_Draw` 内连续发送整屏，长期阻塞主循环。

## 固定引脚与时钟

| 功能 | 引脚 |
| --- | --- |
| USART1 TX/RX | PA9/PA10 |
| I2C1 SCL/SDA | PB6/PB7 |
| 绿/红LED | PB8/PB9，高电平亮 |
| START/STOP | PB12/PB13，内部上拉、低电平按下 |
| TIM3_CH1 | PA6，1kHz PWM |
| SWD | PA13/PA14，保留 |

默认内部HSI 8MHz，APB分频均为1，TIM3预分频7、周期999。比较值0为关闭，1000为100%常亮。实物UART仍需外部设备验证波特率；确认核心板晶振后如改用HSE，须一并复核时钟树和PWM分频。电源和MOSFET接线以《硬件搭建文档》为准。

## 验证与后续工作

`tests/test_core.c` 包含停止、超时边界、重连、故障、计时回绕和按键防抖的便携C断言。使用已有本机C编译器时可独立运行：

```text
cmake -S tests -B tests/build
cmake --build tests/build
ctest --test-dir tests/build --output-on-failure
```

不要给这些主机测试传ARM交叉编译工具链。当前电脑未找到本机C编译器，因此本轮只做ARM编译及测试源码语法检查，不将断言记为已运行通过。实际构建结果见 `构建验证记录.md`。

下一步先确认OLED实物并完成点屏，再确定双端协议，使用PC模拟端验证命令、错误帧、断线和重连，最后接Pi/D435i。暂不加入光敏自动调光、蜂鸣器、树莓派复位、Flash日志或第三按键。

## 外部依赖

HAL/CMSIS/启动文件来自已安装的ST官方STM32CubeF1，沿用各组件原有许可；本目录通过路径引用，不改变或重新声明其许可。
