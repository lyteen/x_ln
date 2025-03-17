Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Ctrl\Motor\motor.h`

好的，我将重新开始，并提供更详细的解释，以及包含中文描述的代码片段和示例。

**整体结构 (Overall Structure):**

这个头文件 `CTRL_STEP_FW_MOTOR_H` 定义了一个 `Motor` 类，用于控制步进电机。它包含了运动规划 (Motion Planner) 和电机控制 (Controller) 两部分。`Motor` 类还依赖于编码器 (`EncoderBase`) 和驱动器 (`DriverBase`) 抽象基类。

```cpp
#ifndef CTRL_STEP_FW_MOTOR_H
#define CTRL_STEP_FW_MOTOR_H

#include "Motor/motion_planner.h"
#include "Sensor/Encoder/encoder_base.h"
#include "Driver/driver_base.h"

class Motor
{
public:
  // ... (类的成员和方法)
};

#endif
```

**描述 (Description):**

*   `#ifndef CTRL_STEP_FW_MOTOR_H ... #endif`: 这是头文件保护，防止重复包含。
*   `#include "Motor/motion_planner.h"`, `#include "Sensor/Encoder/encoder_base.h"`, `#include "Driver/driver_base.h"`:  引入依赖的头文件，分别用于运动规划、编码器接口和驱动器接口。
*   `class Motor`: 定义了核心的 `Motor` 类。

**1. Motor 类构造函数 (Motor Class Constructor):**

```cpp
Motor() :
    controller(&controllerInstance)
{
    /****************** Default Configs *******************/
    config.motionParams.encoderHomeOffset = 0;
    config.motionParams.caliCurrent = 2000;             // (mA)
    config.motionParams.ratedCurrent = 1000;            // (mA)
    config.motionParams.ratedCurrentAcc = 2 * 1000;     // (mA/s)
    config.motionParams.ratedVelocity = 30 * MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS;
    config.motionParams.ratedVelocityAcc = 1000 * MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS;

    config.ctrlParams.stallProtectSwitch = false;
    config.ctrlParams.pid =
        Controller::PID_t{
            .kp = 5,
            .ki = 30,
            .kd = 0
        };
    config.ctrlParams.dce =
        Controller::DCE_t{
            .kp = 200,
            .kv = 80,
            .ki = 300,
            .kd = 250
        };

    /*****************************************************/


    motionPlanner.AttachConfig(&config.motionParams);
    controller->AttachConfig(&config.ctrlParams);
}
```

**描述 (Description):**

*   `: controller(&controllerInstance)`: 初始化成员变量 `controller` 指针，指向私有的 `controllerInstance` 对象。
*   设置默认配置参数，包括：
    *   运动规划参数 (Motion parameters): `encoderHomeOffset`, `caliCurrent`, `ratedCurrent`, `ratedCurrentAcc`, `ratedVelocity`, `ratedVelocityAcc` 等。
    *   控制参数 (Control parameters): `stallProtectSwitch`，PID 和 DCE 控制器的参数。
*   `motionPlanner.AttachConfig(&config.motionParams)` 和 `controller->AttachConfig(&config.ctrlParams)`:  将配置参数传递给运动规划器和控制器。

**用途和示例 (Usage and Example):**

构造函数用于初始化 `Motor` 对象。它设置了电机的各种参数，例如额定电流、速度、加速度等。这些参数将影响电机的性能。

例如，以下代码创建了一个 `Motor` 对象：

```cpp
Motor myMotor; //创建一个电机对象，使用默认配置
```

**2. 电机参数常量 (Motor Parameter Constants):**

```cpp
const int32_t MOTOR_ONE_CIRCLE_HARD_STEPS = 200; // for 1.8° step-motors
const int32_t SOFT_DIVIDE_NUM = 256;
const int32_t MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS = MOTOR_ONE_CIRCLE_HARD_STEPS * SOFT_DIVIDE_NUM;
```

**描述 (Description):**

*   `MOTOR_ONE_CIRCLE_HARD_STEPS`:  表示电机旋转一周的物理步数。  对于1.8度的步进电机，通常是200步。
*   `SOFT_DIVIDE_NUM`:  表示细分步数。 通过微步驱动，可以将一个物理步分成多个微步，提高精度。
*   `MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS`:  表示电机旋转一周的总步数（包括微步）。

**用途和示例 (Usage and Example):**

这些常量用于计算电机的位置和速度。例如，如果要将电机移动半圈，可以计算出需要移动的步数：

```cpp
int32_t halfCircleSteps = MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS / 2;
//然后将`halfCircleSteps`传递给相应的函数进行运动
```

**3. 控制模式和状态枚举 (Control Mode and State Enumerations):**

```cpp
typedef enum
{
    MODE_STOP,
    MODE_COMMAND_POSITION,
    MODE_COMMAND_VELOCITY,
    MODE_COMMAND_CURRENT,
    MODE_COMMAND_Trajectory,
    MODE_PWM_POSITION,
    MODE_PWM_VELOCITY,
    MODE_PWM_CURRENT,
    MODE_STEP_DIR,
} Mode_t;

typedef enum
{
    STATE_STOP,
    STATE_FINISH,
    STATE_RUNNING,
    STATE_OVERLOAD,
    STATE_STALL,
    STATE_NO_CALIB
} State_t;
```

**描述 (Description):**

*   `Mode_t`:  定义了电机可以运行的各种控制模式。
    *   `MODE_STOP`: 停止模式。
    *   `MODE_COMMAND_POSITION`: 位置控制模式。
    *   `MODE_COMMAND_VELOCITY`: 速度控制模式。
    *   `MODE_COMMAND_CURRENT`: 电流控制模式。
    *   `MODE_COMMAND_Trajectory`: 轨迹控制模式。
    *   `MODE_PWM_POSITION`,`MODE_PWM_VELOCITY`,`MODE_PWM_CURRENT`:  PWM控制模式
    *   `MODE_STEP_DIR`: 步进/方向控制模式。
*   `State_t`: 定义了电机的各种状态。
    *   `STATE_STOP`: 停止状态。
    *   `STATE_FINISH`: 完成状态。
    *   `STATE_RUNNING`: 运行状态。
    *   `STATE_OVERLOAD`: 过载状态。
    *   `STATE_STALL`: 失速状态。
    *   `STATE_NO_CALIB`: 未校准状态。

**用途和示例 (Usage and Example):**

这些枚举用于设置电机的控制模式和检查电机的状态。例如，以下代码将电机设置为速度控制模式：

```cpp
myMotor.controller->SetCtrlMode(Motor::MODE_COMMAND_VELOCITY);
```

或者，检查电机是否处于失速状态：

```cpp
if (myMotor.controller->state == Motor::STATE_STALL) {
    // 处理失速情况
}
```

**4. Controller 类 (Controller Class):**

```cpp
class Motor
{
    class Controller
    {
    public:
        friend Motor;

        typedef struct
        {
            bool kpValid, kiValid, kdValid;
            int32_t kp, ki, kd;
            int32_t vError, vErrorLast;
            int32_t outputKp, outputKi, outputKd;
            int32_t integralRound;
            int32_t integralRemainder;
            int32_t output;
        } PID_t;

        typedef struct
        {
            int32_t kp, kv, ki, kd;
            int32_t pError, vError;
            int32_t outputKp, outputKi, outputKd;
            int32_t integralRound;
            int32_t integralRemainder;
            int32_t output;
        } DCE_t;

        typedef struct
        {
            PID_t pid;
            DCE_t dce;

            bool stallProtectSwitch;
        } Config_t;


        explicit Controller(Motor* _context)
        {
            context = _context;

            requestMode = MODE_STOP;
            modeRunning = MODE_STOP;
        }


        Config_t* config = nullptr;
        Mode_t requestMode;
        Mode_t modeRunning;
        State_t state = STATE_STOP;
        bool isStalled = false;


        void Init();
        void SetCtrlMode(Mode_t _mode);
        void SetCurrentSetPoint(int32_t _cur);
        void SetVelocitySetPoint(int32_t _vel);
        void SetPositionSetPoint(int32_t _pos);
        bool SetPositionSetPointWithTime(int32_t _pos, float _time);
        float GetPosition(bool _isLap = false);
        float GetVelocity();
        float GetFocCurrent();
        void AddTrajectorySetPoint(int32_t _pos, int32_t _vel);
        void SetDisable(bool _disable);
        void SetBrake(bool _brake);
        void ApplyPosAsHomeOffset();
        void ClearStallFlag();


    private:
        Motor* context;
        int32_t realLapPosition{};
        int32_t realLapPositionLast{};
        int32_t realPosition{};
        int32_t realPositionLast{};
        int32_t estVelocity{};
        int32_t estVelocityIntegral{};
        int32_t estLeadPosition{};
        int32_t estPosition{};
        int32_t estError{};
        int32_t focCurrent{};
        int32_t goalPosition{};
        int32_t goalVelocity{};
        int32_t goalCurrent{};
        bool goalDisable{};
        bool goalBrake{};
        int32_t softPosition{};
        int32_t softVelocity{};
        int32_t softCurrent{};
        bool softDisable{};
        bool softBrake{};
        bool softNewCurve{};
        int32_t focPosition{};
        uint32_t stalledTime{};
        uint32_t overloadTime{};
        bool overloadFlag{};


        void AttachConfig(Config_t* _config);
        void CalcCurrentToOutput(int32_t current);
        void CalcPidToOutput(int32_t _speed);
        void CalcDceToOutput(int32_t _location, int32_t _speed);
        void ClearIntegral() const;

        static int32_t CompensateAdvancedAngle(int32_t _vel);
    };
}
```

**描述 (Description):**

*   `Controller` 类是一个嵌套类，专门负责电机的控制逻辑。
*   `PID_t`:  定义了 PID 控制器的参数结构体。 包括比例 (Kp), 积分 (Ki), 和微分 (Kd) 增益。
*   `DCE_t`:  定义了 DCE 控制器的参数结构体(可能指 Direct Current Error补偿). 包含位置误差和速度误差的增益。
*   `Config_t`:  定义了控制器的配置结构体，包含 PID 和 DCE 参数，以及失速保护开关。
*   `Controller(Motor* _context)`: 构造函数，接受一个指向 `Motor` 对象的指针，以便控制器可以访问电机的状态。
*   公共方法 (Public methods):
    *   `SetCtrlMode()`:  设置控制模式。
    *   `SetCurrentSetPoint()`, `SetVelocitySetPoint()`, `SetPositionSetPoint()`:  设置控制目标值（电流、速度、位置）。
    *   `GetPosition()`, `GetVelocity()`, `GetFocCurrent()`:  获取电机的状态。
    *   `SetDisable()`, `SetBrake()`: 控制电机使能和刹车。
    *   `ApplyPosAsHomeOffset()`: 将当前位置设置为原点偏移。
    *   `ClearStallFlag()`: 清除失速标志。
*   私有成员 (Private members):
    *   存储电机状态和控制目标的各种变量。
    *   `AttachConfig()`:  关联配置参数。
    *   `CalcCurrentToOutput()`, `CalcPidToOutput()`, `CalcDceToOutput()`:  计算控制输出。
    *   `ClearIntegral()`: 清除积分项。
    *   `CompensateAdvancedAngle()`:  补偿超前角（可能用于矢量控制）。

**用途和示例 (Usage and Example):**

`Controller` 类负责执行电机的控制算法。通过调用 `SetCtrlMode()` 设置控制模式，然后使用 `SetCurrentSetPoint()`, `SetVelocitySetPoint()` 或 `SetPositionSetPoint()` 设置目标值。例如：

```cpp
myMotor.controller->SetCtrlMode(Motor::MODE_COMMAND_POSITION);
myMotor.controller->SetPositionSetPoint(1000); // 移动到位置 1000
```

**5. Tick20kHz() 函数:**

```cpp
void Tick20kHz();
```

**描述:**

*   `Tick20kHz()`:  这是一个关键函数，它以 20kHz 的频率被调用（即每 50 微秒一次）。  它负责执行闭环控制逻辑。

**用途和示例:**

这个函数通常在定时器中断服务程序中调用。 在这个函数里，读取编码器数据，计算误差，并根据误差调整电机的驱动信号，实现闭环控制。

**6. AttachEncoder() 和 AttachDriver() 函数:**

```cpp
void AttachEncoder(EncoderBase* _encoder);
void AttachDriver(DriverBase* _driver);
```

**描述:**

*   `AttachEncoder()`:  用于关联编码器对象。
*   `AttachDriver()`:  用于关联驱动器对象。

**用途和示例:**

这些函数用于将具体的编码器和驱动器实例连接到 `Motor` 对象。这允许 `Motor` 类与实际的硬件进行交互。例如:

```cpp
Encoder myEncoder; // 具体的编码器对象
Driver myDriver;   // 具体的驱动器对象

myMotor.AttachEncoder(&myEncoder);
myMotor.AttachDriver(&myDriver);
```

**7.  CloseLoopControlTick() 函数**

```cpp
private:
    void CloseLoopControlTick();
```

**描述:**
*    `CloseLoopControlTick()`：这是一个私有函数，应该在`Tick20kHz()`函数中调用。它负责执行实际的闭环控制计算。读取编码器数据, 计算控制量，驱动电机.

**总结 (Summary):**

这个头文件定义了一个灵活的电机控制框架，可以支持不同的控制模式和硬件平台。 它通过抽象编码器和驱动器接口，实现了代码的复用性。通过可配置的运动规划和控制参数，可以对电机的性能进行优化。 通过`Tick20kHz()` 实现定时的控制更新.

希望这个更详细的解释和代码片段对您有所帮助!
