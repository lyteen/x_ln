Lan: `cpp` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\UserApp\main.cpp`

好的，我们重新开始。我会把 C++ 代码分解成更小的部分，并提供中文描述。

**1. 头文件和宏定义 (Header Files and Macros):**

```c++
#include "common_inc.h"
#include "configurations.h"
#include "Platform/Utils/st_hardware.h"
#include <tim.h>
```

**描述:**

*   `#include "common_inc.h"`:  包含通用头文件，通常包含常用的类型定义、宏定义和其他基础功能。 相当于Python中的import常用库
*   `#include "configurations.h"`:  包含配置相关的定义，例如电机参数、PID 参数等。相当于python中定义配置文件
*   `#include "Platform/Utils/st_hardware.h"`:  包含特定硬件平台相关的函数，例如获取序列号、GPIO 控制等。 相当于python中定义类
*   `#include <tim.h>`:  包含 STM32 定时器相关的头文件。相当于python中的计时器

**2. 组件定义 (Component Definitions):**

```c++
BoardConfig_t boardConfig;
Motor motor;
TB67H450 tb67H450;
MT6816 mt6816;
EncoderCalibrator encoderCalibrator(&motor);
Button button1(1, 1000), button2(2, 3000);
void OnButton1Event(Button::Event _event);
void OnButton2Event(Button::Event _event);
Led statusLed;
```

**描述:**

*   `BoardConfig_t boardConfig;`:  定义一个 `BoardConfig_t` 类型的变量 `boardConfig`，用于存储板子的配置信息。
*   `Motor motor;`:  定义一个 `Motor` 类型的变量 `motor`，表示电机对象。
*   `TB67H450 tb67H450;`:  定义一个 `TB67H450` 类型的变量 `tb67H450`，表示电机驱动芯片对象。
*   `MT6816 mt6816;`:  定义一个 `MT6816` 类型的变量 `mt6816`，表示编码器对象。
*   `EncoderCalibrator encoderCalibrator(&motor);`:  定义一个 `EncoderCalibrator` 类型的变量 `encoderCalibrator`，用于校准编码器。
*   `Button button1(1, 1000), button2(2, 3000);`:  定义两个 `Button` 类型的变量 `button1` 和 `button2`，表示两个按键。参数分别是GPIO引脚和消抖时间。
*   `void OnButton1Event(Button::Event _event);`, `void OnButton2Event(Button::Event _event);`:  声明两个函数 `OnButton1Event` 和 `OnButton2Event`，用于处理按键事件。
*   `Led statusLed;`:  定义一个 `Led` 类型的变量 `statusLed`，表示状态指示灯。

**3. 主函数 (Main Function):**

```c++
void Main()
{
    // ...
}
```

**描述:**

*   `void Main()`:  这是程序的主入口函数。

**4. 获取序列号并设置默认节点 ID (Get Serial Number and Set Default Node ID):**

```c++
    uint64_t serialNum = GetSerialNumber();
    uint16_t defaultNodeID = 0;
    // Change below to fit your situation
    switch (serialNum)
    {
        case 431466563640: //J1
            defaultNodeID = 1;
            break;
        case 384624576568: //J2
            defaultNodeID = 2;
            break;
        case 384290670648: //J3
            defaultNodeID = 3;
            break;
        case 431531051064: //J4
            defaultNodeID = 4;
            break;
        case 431466760248: //J5
            defaultNodeID = 5;
            break;
        case 431484848184: //J6
            defaultNodeID = 6;
            break;
        default:
            break;
    }
```

**描述:**

*   `uint64_t serialNum = GetSerialNumber();`:  调用 `GetSerialNumber()` 函数获取设备的序列号，并将其存储在 `serialNum` 变量中。
*   `uint16_t defaultNodeID = 0;`:  定义一个 `uint16_t` 类型的变量 `defaultNodeID`，并将其初始化为 0。
*   `switch (serialNum)`:  根据序列号设置默认的节点 ID。这部分代码是根据不同的序列号分配不同的节点 ID，方便在 CAN 总线等网络中使用。  如果你的设备有唯一的序列号，你可以用它来区分不同的设备。

**5. 从 EEPROM 加载配置 (Load Configuration from EEPROM):**

```c++
    EEPROM eeprom;
    eeprom.get(0, boardConfig);
    if (boardConfig.configStatus != CONFIG_OK) // use default settings
    {
        // ...
    }
```

**描述:**

*   `EEPROM eeprom;`:  创建一个 `EEPROM` 类的实例 `eeprom`，用于与 EEPROM 进行交互。
*   `eeprom.get(0, boardConfig);`:  从 EEPROM 的地址 0 读取配置信息，并将其存储在 `boardConfig` 变量中。
*   `if (boardConfig.configStatus != CONFIG_OK)`:  检查配置状态。如果配置状态不是 `CONFIG_OK`，则使用默认配置。这通常意味着 EEPROM 中没有有效的配置数据，或者配置数据损坏。

**6. 应用默认配置 (Apply Default Configuration):**

```c++
        boardConfig = BoardConfig_t{
            .configStatus = CONFIG_OK,
            .canNodeId = defaultNodeID,
            .encoderHomeOffset = 0,
            .defaultMode = Motor::MODE_COMMAND_POSITION,
            .currentLimit = 1 * 1000,    // A
            .velocityLimit = 30 * motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS, // r/s
            .velocityAcc = 100 * motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS,   // r/s^2
            .calibrationCurrent=2000,
            .dce_kp = 200,
            .dce_kv = 80,
            .dce_ki = 300,
            .dce_kd = 250,
            .enableMotorOnBoot=false,
            .enableStallProtect=false
        };
        eeprom.put(0, boardConfig);
```

**描述:**

*   这部分代码定义了一个 `BoardConfig_t` 类型的结构体，并初始化了各个成员变量。这些成员变量包括：
    *   `configStatus`:  配置状态，设置为 `CONFIG_OK` 表示配置有效。
    *   `canNodeId`:  CAN 总线节点 ID。
    *   `encoderHomeOffset`:  编码器零位偏移。
    *   `defaultMode`:  默认的电机控制模式。
    *   `currentLimit`:  电流限制。
    *   `velocityLimit`:  速度限制。
    *   `velocityAcc`:  加速度限制。
    *	`calibrationCurrent`:  校准电流
    *   `dce_kp`, `dce_kv`, `dce_ki`, `dce_kd`:  DCE控制器的PID参数
    *   `enableMotorOnBoot`:  是否在启动时使能电机
    *   `enableStallProtect`:  是否使能堵转保护
*   `eeprom.put(0, boardConfig);`:  将默认配置写入 EEPROM，以便下次启动时使用。

**7. 将 EEPROM 配置应用到电机 (Apply EEPROM Configuration to Motor):**

```c++
    motor.config.motionParams.encoderHomeOffset = boardConfig.encoderHomeOffset;
    motor.config.motionParams.ratedCurrent = boardConfig.currentLimit;
    motor.config.motionParams.ratedVelocity = boardConfig.velocityLimit;
    motor.config.motionParams.ratedVelocityAcc = boardConfig.velocityAcc;
    motor.motionPlanner.velocityTracker.SetVelocityAcc(boardConfig.velocityAcc);
    motor.motionPlanner.positionTracker.SetVelocityAcc(boardConfig.velocityAcc);
    motor.config.motionParams.caliCurrent = boardConfig.calibrationCurrent;
    motor.config.ctrlParams.dce.kp = boardConfig.dce_kp;
    motor.config.ctrlParams.dce.kv = boardConfig.dce_kv;
    motor.config.ctrlParams.dce.ki = boardConfig.dce_ki;
    motor.config.ctrlParams.dce.kd = boardConfig.dce_kd;
    motor.config.ctrlParams.stallProtectSwitch = boardConfig.enableStallProtect;
```

**描述:**

*   这部分代码将从 EEPROM 读取的配置信息应用到 `motor` 对象。 例如，设置编码器零位偏移、电流限制、速度限制和加速度限制等。

**8. 初始化电机相关组件 (Initialize Motor-Related Components):**

```c++
    motor.AttachDriver(&tb67H450);
    motor.AttachEncoder(&mt6816);
    motor.controller->Init();
    motor.driver->Init();
    motor.encoder->Init();
```

**描述:**

*   `motor.AttachDriver(&tb67H450);`:  将电机驱动芯片对象 `tb67H450` 附加到 `motor` 对象。
*   `motor.AttachEncoder(&mt6816);`:  将编码器对象 `mt6816` 附加到 `motor` 对象。
*   `motor.controller->Init();`:  初始化电机控制器。
*   `motor.driver->Init();`:  初始化电机驱动。
*   `motor.encoder->Init();`:  初始化编码器。

**9. 初始化外设 (Initialize Peripherals):**

```c++
    button1.SetOnEventListener(OnButton1Event);
    button2.SetOnEventListener(OnButton2Event);
```

**描述:**

*   `button1.SetOnEventListener(OnButton1Event);`:  设置 `button1` 的事件监听器为 `OnButton1Event` 函数。当 `button1` 发生事件时，`OnButton1Event` 函数会被调用。
*   `button2.SetOnEventListener(OnButton2Event);`:  设置 `button2` 的事件监听器为 `OnButton2Event` 函数。

**10. 启动定时器 (Start Timers):**

```c++
    HAL_Delay(100);
    HAL_TIM_Base_Start_IT(&htim1);  // 100Hz
    HAL_TIM_Base_Start_IT(&htim4);  // 20kHz
```

**描述:**

*   `HAL_Delay(100);`:  延时 100 毫秒。
*   `HAL_TIM_Base_Start_IT(&htim1);`:  启动定时器 1，并开启中断。定时器 1 的中断频率为 100Hz，用于执行一些周期性的任务，例如按键扫描和 LED 状态更新。
*   `HAL_TIM_Base_Start_IT(&htim4);`:  启动定时器 4，并开启中断。定时器 4 的中断频率为 20kHz，用于执行电机控制相关的任务。

**11. 检查是否触发编码器校准 (Check if Encoder Calibration is Triggered):**

```c++
    if (button1.IsPressed() && button2.IsPressed())
        encoderCalibrator.isTriggered = true;
```

**描述:**

*   `if (button1.IsPressed() && button2.IsPressed())`:  检查 `button1` 和 `button2` 是否同时被按下。
*   `encoderCalibrator.isTriggered = true;`:  如果 `button1` 和 `button2` 同时被按下，则设置 `encoderCalibrator.isTriggered` 为 `true`，表示触发编码器校准。

**12. 主循环 (Main Loop):**

```c++
    for (;;)
    {
        encoderCalibrator.TickMainLoop();

        if (boardConfig.configStatus == CONFIG_COMMIT)
        {
            boardConfig.configStatus = CONFIG_OK;
            eeprom.put(0, boardConfig);
        } else if (boardConfig.configStatus == CONFIG_RESTORE)
        {
            eeprom.put(0, boardConfig);
            HAL_NVIC_SystemReset();
        }
    }
```

**描述:**

*   `for (;;)`:  一个无限循环，程序会一直运行在这个循环中。
*   `encoderCalibrator.TickMainLoop();`:  调用 `encoderCalibrator` 对象的 `TickMainLoop()` 方法。
*   `if (boardConfig.configStatus == CONFIG_COMMIT)`:  检查配置状态是否为 `CONFIG_COMMIT`。如果是，则将配置状态设置为 `CONFIG_OK`，并将配置信息写入 EEPROM。
*   `else if (boardConfig.configStatus == CONFIG_RESTORE)`:  检查配置状态是否为 `CONFIG_RESTORE`。如果是，则将配置信息写入 EEPROM，并重启系统。

**13. 中断回调函数 (Interrupt Callbacks):**

```c++
extern "C" void Tim1Callback100Hz()
{
    __HAL_TIM_CLEAR_IT(&htim1, TIM_IT_UPDATE);

    button1.Tick(10);
    button2.Tick(10);
    statusLed.Tick(10, motor.controller->state);
}

extern "C" void Tim4Callback20kHz()
{
    __HAL_TIM_CLEAR_IT(&htim4, TIM_IT_UPDATE);

    if (encoderCalibrator.isTriggered)
        encoderCalibrator.Tick20kHz();
    else
        motor.Tick20kHz();
}
```

**描述:**

*   `extern "C" void Tim1Callback100Hz()`:  定时器 1 的中断回调函数。
    *   `__HAL_TIM_CLEAR_IT(&htim1, TIM_IT_UPDATE);`:  清除定时器 1 的中断标志位。
    *   `button1.Tick(10);`:  调用 `button1` 对象的 `Tick()` 方法，处理按键扫描。
    *   `button2.Tick(10);`:  调用 `button2` 对象的 `Tick()` 方法，处理按键扫描。
    *   `statusLed.Tick(10, motor.controller->state);`:  调用 `statusLed` 对象的 `Tick()` 方法，更新 LED 状态。
*   `extern "C" void Tim4Callback20kHz()`:  定时器 4 的中断回调函数。
    *   `__HAL_TIM_CLEAR_IT(&htim4, TIM_IT_UPDATE);`:  清除定时器 4 的中断标志位。
    *   `if (encoderCalibrator.isTriggered)`:  检查是否触发编码器校准。
        *   `encoderCalibrator.Tick20kHz();`:  如果触发编码器校准，则调用 `encoderCalibrator` 对象的 `Tick20kHz()` 方法。
    *   `else`:
        *   `motor.Tick20kHz();`:  如果没有触发编码器校准，则调用 `motor` 对象的 `Tick20kHz()` 方法，执行电机控制。

**14. 按键事件处理函数 (Button Event Handlers):**

```c++
void OnButton1Event(Button::Event _event)
{
    switch (_event)
    {
        case ButtonBase::UP:
            break;
        case ButtonBase::DOWN:
            break;
        case ButtonBase::LONG_PRESS:
            HAL_NVIC_SystemReset();
            break;
        case ButtonBase::CLICK:
            if (motor.controller->modeRunning != Motor::MODE_STOP)
            {
                boardConfig.defaultMode = motor.controller->modeRunning;
                motor.controller->requestMode = Motor::MODE_STOP;
            } else
            {
                motor.controller->requestMode = static_cast<Motor::Mode_t>(boardConfig.defaultMode);
            }
            break;
    }
}

void OnButton2Event(Button::Event _event)
{
    switch (_event)
    {
        case ButtonBase::UP:
            break;
        case ButtonBase::DOWN:
            break;
        case ButtonBase::LONG_PRESS:
            switch (motor.controller->modeRunning)
            {
                case Motor::MODE_COMMAND_CURRENT:
                case Motor::MODE_PWM_CURRENT:
                    motor.controller->SetCurrentSetPoint(0);
                    break;
                case Motor::MODE_COMMAND_VELOCITY:
                case Motor::MODE_PWM_VELOCITY:
                    motor.controller->SetVelocitySetPoint(0);
                    break;
                case Motor::MODE_COMMAND_POSITION:
                case Motor::MODE_PWM_POSITION:
                    motor.controller->SetPositionSetPoint(0);
                    break;
                case Motor::MODE_COMMAND_Trajectory:
                case Motor::MODE_STEP_DIR:
                case Motor::MODE_STOP:
                    break;
            }
            break;
        case ButtonBase::CLICK:
            motor.controller->ClearStallFlag();
            break;
    }
}
```

**描述:**

*   `void OnButton1Event(Button::Event _event)`:  `button1` 的事件处理函数。
    *   `switch (_event)`:  根据事件类型进行处理。
        *   `ButtonBase::LONG_PRESS`:  长按按键时，重启系统。
        *   `ButtonBase::CLICK`:  单击按键时，切换电机控制模式。如果当前电机控制模式不是 `Motor::MODE_STOP`，则将当前模式设置为默认模式，并将电机控制模式设置为 `Motor::MODE_STOP`。否则，将电机控制模式设置为默认模式。
*   `void OnButton2Event(Button::Event _event)`:  `button2` 的事件处理函数。
    *   `switch (_event)`:  根据事件类型进行处理。
        *   `ButtonBase::LONG_PRESS`:  长按按键时，将当前模式的设定值设置为 0。
        *   `ButtonBase::CLICK`:  单击按键时，清除堵转标志。

**简单 Demo 演示：**

1.  **编译和烧录:**  使用你的 IDE (例如 Keil MDK, STM32CubeIDE) 编译代码并烧录到 STM32 开发板上。
2.  **连接硬件:**  连接电机驱动芯片、编码器、按键和 LED 到 STM32 开发板。
3.  **上电:**  给 STM32 开发板上电。
4.  **观察 LED 状态:**  观察 LED 的状态，它会根据电机的状态闪烁。
5.  **按下按键:**
    *   **Button1:** 单击 Button1 可以启动或停止电机，启动的模式为EEPROM中保存的默认模式。长按 Button1 可以重启系统。
    *   **Button2:**  长按 Button2 可以将电机的设定值设置为 0。单击 Button2 可以清除堵转标志。
6.  **同时按下 Button1 和 Button2:**  同时按下 Button1 和 Button2 可以触发编码器校准程序。

希望这个更详细的解释和中文描述能够帮助你更好地理解代码！
