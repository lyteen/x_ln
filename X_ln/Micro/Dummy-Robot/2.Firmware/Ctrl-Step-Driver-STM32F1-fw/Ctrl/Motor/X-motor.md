Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Ctrl\Motor\motor.h`

Okay, let's refactor and improve this C++ motor control code. I'll break it down into smaller, more manageable pieces with explanations in both English and Chinese.  I'll focus on improvements like error handling, clearer state management, and potentially adding a command queue.

**1.  `Motor.h` (Main Header File)**

```c++
#ifndef CTRL_STEP_FW_MOTOR_H
#define CTRL_STEP_FW_MOTOR_H

#include <stdint.h> // Required for int32_t, uint32_t
#include <stdbool.h> // Required for bool

#include "Motor/motion_planner.h"
#include "Sensor/Encoder/encoder_base.h"
#include "Driver/driver_base.h"

// Forward declarations (避免循环依赖)
class Controller;

class Motor
{
public:
    Motor();
    ~Motor(); // Destructor

    // Constants
    static const int32_t MOTOR_ONE_CIRCLE_HARD_STEPS; // for 1.8° step-motors
    static const int32_t SOFT_DIVIDE_NUM;
    static const int32_t MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS;

    // Types
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
        MODE_INVALID  // Add an invalid state
    } Mode_t;

    typedef enum
    {
        STATE_STOP,
        STATE_FINISH,
        STATE_RUNNING,
        STATE_OVERLOAD,
        STATE_STALL,
        STATE_NO_CALIB,
        STATE_ERROR // Add an error state
    } State_t;


    // Public API
    void Tick20kHz();
    void AttachEncoder(EncoderBase* _encoder);
    void AttachDriver(DriverBase* _driver);

    // Accessors (Getters)  (访问器)
    State_t GetState() const { return state; }
    Mode_t GetMode() const { return currentMode; }

    // Configuration
    struct Config_t
    {
        MotionPlanner::Config_t motionParams{};
        Controller::Config_t ctrlParams{};
    };
    Config_t config;

private:
    // Private Members
    Controller* controller;
    MotionPlanner motionPlanner;
    EncoderBase* encoder;
    DriverBase* driver;
    State_t state;
    Mode_t currentMode;

    // Private Methods
    void CloseLoopControlTick();

    // Helper function to handle errors more gracefully
    void HandleError(const char* message); //添加了错误处理函数

    // Static member (内部静态成员)
    static Controller controllerInstance; // Moved here


};

#endif
```

**Explanation:**

*   **Includes:**  Standard includes for `stdint.h` and `stdbool.h` are added.
*   **Forward Declaration:**  `class Controller;` is a forward declaration.  This is crucial to break circular dependencies.
*   **Destructor:** A destructor `~Motor()` is declared for cleanup (important for resource management, although it's currently empty).
*   **Constants:**  Made the constants `static` to ensure they belong to the class itself and are not instance-specific.  This is generally better practice.
*   **`MODE_INVALID` and `STATE_ERROR`:** Added `MODE_INVALID` and `STATE_ERROR` to the enums to handle invalid states or errors more explicitly.
*   **Accessors:** Added `GetState()` and `GetMode()` to allow external code to safely query the motor's state and mode.  This promotes encapsulation.  These are `const` because they don't modify the object's state.
*   **Private Members:** Moved the `controllerInstance` declaration to the private section.
*   **`HandleError()`:** Added a `HandleError()` function for centralized error handling.
*   **State and Mode:**  Added `state` and `currentMode` as private members.

**Chinese Explanation:**

*   **包含文件:**  添加了 `stdint.h` 和 `stdbool.h` 的标准包含。
*   **前向声明:**  `class Controller;` 是一个前向声明，用于避免循环依赖。
*   **析构函数:**  声明了一个析构函数 `~Motor()`，用于清理资源（虽然目前为空），这对于资源管理很重要。
*   **常量:**  将常量声明为 `static`，以确保它们属于类本身，而不是实例特定的。这通常是更好的做法。
*   **`MODE_INVALID` 和 `STATE_ERROR`:**  在枚举中添加了 `MODE_INVALID` 和 `STATE_ERROR`，以便更明确地处理无效状态或错误。
*   **访问器:**  添加了 `GetState()` 和 `GetMode()`，以允许外部代码安全地查询电机的状态和模式。这促进了封装。这些都是 `const`，因为它们不修改对象的状态。
*   **私有成员:**  将 `controllerInstance` 声明移到私有部分。
*   **`HandleError()`:**  添加了 `HandleError()` 函数，用于集中式错误处理。
*   **状态和模式:**  将 `state` 和 `currentMode` 添加为私有成员。

**2. `Motor.cpp` (Implementation File)**

```c++
#include "Motor.h"
#include <iostream> // For error handling (e.g., printing to console)

// Initialize static members (初始化静态成员)
const int32_t Motor::MOTOR_ONE_CIRCLE_HARD_STEPS = 200;
const int32_t Motor::SOFT_DIVIDE_NUM = 256;
const int32_t Motor::MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS = Motor::MOTOR_ONE_CIRCLE_HARD_STEPS * Motor::SOFT_DIVIDE_NUM;

Controller Motor::controllerInstance = Controller(this); //Initialize static member

Motor::Motor() :
    controller(&controllerInstance),
    encoder(nullptr), // Initialize pointers to null
    driver(nullptr),
    state(STATE_STOP), // Initialize state
    currentMode(MODE_STOP) // Initialize mode
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

Motor::~Motor()
{
    // Clean up resources if needed (例如，释放动态分配的内存)
    // Important if you dynamically allocate encoder/driver.
}


void Motor::Tick20kHz()
{
    if (encoder == nullptr || driver == nullptr)
    {
        HandleError("Encoder or driver not attached!");
        state = STATE_ERROR;
        return;
    }

    if (state == STATE_ERROR) return; // Don't do anything if in error state.

    CloseLoopControlTick();
}


void Motor::AttachEncoder(EncoderBase* _encoder)
{
    if (_encoder == nullptr) {
        HandleError("Attempting to attach a null encoder.");
        return;
    }
    encoder = _encoder;
}

void Motor::AttachDriver(DriverBase* _driver)
{
    if (_driver == nullptr) {
        HandleError("Attempting to attach a null driver.");
        return;
    }
    driver = _driver;
}


void Motor::CloseLoopControlTick()
{
    // TODO: Implement closed-loop control logic here
    // This is where you'd read the encoder, calculate the error,
    // and command the driver.
}


void Motor::HandleError(const char* message)
{
    std::cerr << "Motor Error: " << message << std::endl;
    // You might also log the error, trigger a fault signal, etc.
    state = STATE_ERROR; // Set the error state
}
```

**Explanation:**

*   **Includes:**  Includes the `Motor.h` header.
*   **Static Member Initialization:**  The static members `MOTOR_ONE_CIRCLE_HARD_STEPS`, `SOFT_DIVIDE_NUM`, and `MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` are initialized.  Crucially, `controllerInstance` is initialized here.
*   **Constructor Initialization:** The constructor now explicitly initializes `encoder`, `driver`, `state`, and `currentMode` to sensible defaults (nullptr and `STATE_STOP`, respectively).
*   **Error Handling in `Tick20kHz`:** Added a check to ensure that the encoder and driver are attached *before* attempting to use them.  If they are not, an error is logged, and the `state` is set to `STATE_ERROR`. The function returns early if the motor is in an error state.
*   **Error Handling in `AttachEncoder` and `AttachDriver`:** Added checks to prevent attaching null encoders or drivers.
*   **`HandleError()` Implementation:** The `HandleError()` function now prints an error message to `std::cerr` and sets the motor's state to `STATE_ERROR`.
*   **Destructor Implementation:** Destructor implemented (empty for now).
*   **TODO Comment:** Left a `TODO` comment in `CloseLoopControlTick()` to remind you to implement the actual control logic.

**Chinese Explanation:**

*   **包含文件:**  包含 `Motor.h` 头文件。
*   **静态成员初始化:**  静态成员 `MOTOR_ONE_CIRCLE_HARD_STEPS`，`SOFT_DIVIDE_NUM` 和 `MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 被初始化。 关键的是，`controllerInstance` 在这里被初始化。
*   **构造函数初始化:**  构造函数现在显式地将 `encoder`，`driver`，`state` 和 `currentMode` 初始化为合理的默认值（分别为 nullptr 和 `STATE_STOP`）。
*   **`Tick20kHz` 中的错误处理:**  添加了一个检查，以确保在尝试使用编码器和驱动器 *之前* 先连接它们。 如果没有，则记录错误，并将 `state` 设置为 `STATE_ERROR`。 如果电机处于错误状态，该函数将提前返回。
*   **`AttachEncoder` 和 `AttachDriver` 中的错误处理:**  添加了检查以防止连接空编码器或驱动器。
*   **`HandleError()` 实现:**  `HandleError()` 函数现在将错误消息打印到 `std::cerr`，并将电机的状态设置为 `STATE_ERROR`。
*   **析构函数实现:** 实现析构函数（目前为空）。
*   **TODO 注释:**  在 `CloseLoopControlTick()` 中留下了一个 `TODO` 注释，以提醒您实现实际的控制逻辑。

**3. `Controller.h` (Modified Controller Header)**

```c++
#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <stdint.h>
#include <stdbool.h>
#include "Motor/motion_planner.h" // Include motion_planner.h

class Motor; // Forward declaration for Motor

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


    explicit Controller(Motor* _context);
    ~Controller();

    Config_t* config = nullptr;
    Motor::Mode_t requestMode;
    Motor::Mode_t modeRunning;
    Motor::State_t state;
    bool isStalled = false;


    void Init();
    void SetCtrlMode(Motor::Mode_t _mode);
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

#endif
```

**Key Changes in `Controller.h`:**

*   **Forward Declaration:** Added `class Motor;` at the beginning to avoid a circular dependency.
*   **Using `Motor::Mode_t` and `Motor::State_t`:**  The `Controller` now uses `Motor::Mode_t` and `Motor::State_t` to ensure type consistency and avoid potential conflicts.
*   **Destructor Declaration:** Added a destructor `~Controller()`.

**Chinese Key Changes:**

*   **前向声明:** 在开头添加了 `class Motor;` 以避免循环依赖。
*   **使用 `Motor::Mode_t` 和 `Motor::State_t`:** `Controller` 现在使用 `Motor::Mode_t` 和 `Motor::State_t` 以确保类型一致性并避免潜在的冲突。
*   **析构函数声明:** 添加了一个析构函数 `~Controller()`。

**4. `Controller.cpp` (Modified Controller Implementation)**

```c++
#include "Controller.h"
#include "Motor.h"  // Include Motor.h, now safe because of forward declaration
#include <iostream>

Controller::Controller(Motor* _context) :
    context(_context),
    requestMode(Motor::MODE_STOP), // Use Motor::MODE_STOP
    modeRunning(Motor::MODE_STOP), // Use Motor::MODE_STOP
    state(Motor::STATE_STOP)     // Use Motor::STATE_STOP
{
}

Controller::~Controller()
{
  //Cleanup
}

void Controller::Init()
{
    // Initialization logic
}


void Controller::SetCtrlMode(Motor::Mode_t _mode)
{
    if (_mode >= Motor::MODE_INVALID) {
        std::cerr << "Invalid control mode requested." << std::endl;
        return;
    }
    requestMode = _mode;
}


void Controller::AttachConfig(Config_t* _config)
{
    config = _config;
}

// Dummy implementations for other methods
void Controller::SetCurrentSetPoint(int32_t _cur) { /* ... */ }
void Controller::SetVelocitySetPoint(int32_t _vel) { /* ... */ }
void Controller::SetPositionSetPoint(int32_t _pos) { /* ... */ }
bool Controller::SetPositionSetPointWithTime(int32_t _pos, float _time) { return false; }
float Controller::GetPosition(bool _isLap) { return 0.0f; }
float Controller::GetVelocity() { return 0.0f; }
float Controller::GetFocCurrent() { return 0.0f; }
void Controller::AddTrajectorySetPoint(int32_t _pos, int32_t _vel) { /* ... */ }
void Controller::SetDisable(bool _disable) { /* ... */ }
void Controller::SetBrake(bool _brake) { /* ... */ }
void Controller::ApplyPosAsHomeOffset() { /* ... */ }
void Controller::ClearStallFlag() { /* ... */ }
void Controller::CalcCurrentToOutput(int32_t current) { /* ... */ }
void Controller::CalcPidToOutput(int32_t _speed) { /* ... */ }
void Controller::CalcDceToOutput(int32_t _location, int32_t _speed) { /* ... */ }
void Controller::ClearIntegral() const { /* ... */ }
int32_t Controller::CompensateAdvancedAngle(int32_t _vel) { return 0; }
```

**Key Changes in `Controller.cpp`:**

*   **Include `Motor.h`:** Now includes `Motor.h`, which is safe because of the forward declaration in `Controller.h`.
*   **Constructor Initialization:**  The constructor initializes `requestMode`, `modeRunning`, and `state` to `Motor::MODE_STOP` and `Motor::STATE_STOP`, respectively.  This ensures proper initialization.
*   **Mode Validation:** Added a check in `SetCtrlMode` to validate the requested mode.
*   **Dummy Implementations:** The other methods have dummy implementations (marked with `/* ... */`) to allow the code to compile.  You'll need to replace these with the real logic.
*   **Destructor Implementation:** Destructor implemented (empty for now).

**Chinese Key Changes:**

*   **包含 `Motor.h`:** 现在包含 `Motor.h`，由于 `Controller.h` 中的前向声明，这是安全的。
*   **构造函数初始化:** 构造函数将 `requestMode`，`modeRunning` 和 `state` 初始化为 `Motor::MODE_STOP` 和 `Motor::STATE_STOP`。 这确保了正确的初始化。
*   **模式验证:** 在 `SetCtrlMode` 中添加了检查以验证请求的模式。
*   **虚拟实现:** 其他方法具有虚拟实现（标记为 `/* ... */`），以允许代码编译。 您需要用实际逻辑替换这些。
*    **析构函数实现:** 实现析构函数（目前为空）。

**5. Simple Usage Example:**

```c++
#include "Motor.h"
#include <iostream>

int main() {
    Motor myMotor;

    // Simulate attaching an encoder and driver (replace with actual instances)
    // Assuming you have concrete EncoderBase and DriverBase classes
    // For example:
    // EncoderBase* myEncoder = new ConcreteEncoder();
    // DriverBase* myDriver = new ConcreteDriver();
    // myMotor.AttachEncoder(myEncoder);
    // myMotor.AttachDriver(myDriver);

    // Set a control mode
    myMotor.controller->SetCtrlMode(Motor::MODE_COMMAND_POSITION);

    // Run the motor control loop for a few cycles
    for (int i = 0; i < 100; ++i) {
        myMotor.Tick20kHz();
        // Simulate some encoder input (replace with actual encoder readings)
        // myEncoder->SimulateTick();
    }

    std::cout << "Motor State: " << myMotor.GetState() << std::endl;
    std::cout << "Motor Mode: " << myMotor.GetMode() << std::endl;

    // Clean up (重要：释放分配的内存!)
    // delete myEncoder;
    // delete myDriver;

    return 0;
}
```

**Important Considerations and Next Steps:**

*   **Resource Management:**  Pay close attention to resource management, especially if you're dynamically allocating `EncoderBase` and `DriverBase` objects.  Use `new` and `delete` carefully.  Consider using smart pointers (e.g., `std::unique_ptr`) to automate memory management.
*   **Real Implementations:** You'll need to provide *real* implementations for the `EncoderBase`, `DriverBase`, `MotionPlanner`, and `Controller` classes.  The dummy implementations are just placeholders.
*   **Concurrency:**  If your motor control loop needs to run in a separate thread, you'll need to add appropriate synchronization mechanisms (e.g., mutexes) to protect shared data.
*   **Command Queue:** Consider adding a command queue to the `Motor` class to allow you to queue up multiple commands.  This would make the motor control more flexible and responsive.
*   **Error Handling:**  Expand the error handling to log errors, trigger fault signals, and potentially attempt to recover from errors.
*   **Testing:**  Write unit tests to verify that the motor control logic is working correctly.

This revised structure provides a more robust and maintainable foundation for your motor control system.  Remember to fill in the missing implementations and add the necessary error handling and synchronization.
