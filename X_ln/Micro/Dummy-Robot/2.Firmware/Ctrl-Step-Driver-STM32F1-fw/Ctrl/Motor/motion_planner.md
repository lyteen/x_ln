Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Ctrl\Motor\motion_planner.h`

Okay, I will provide a fresh explanation of the C++ header file, breaking it down into smaller, more understandable parts with Chinese descriptions and usage examples.

**File Overview:**

The header file `CTRL_STEP_FW_MOTION_PLANNER_H` defines a `MotionPlanner` class. This class is designed to manage the motion control of a system, likely a motor or actuator, by tracking and adjusting current, velocity, and position. It uses several nested classes (CurrentTracker, VelocityTracker, etc.) to handle specific aspects of the motion planning process.  The `#ifndef`, `#define`, and `#endif` directives ensure that the header file is included only once during compilation, preventing errors.

**1. Header Guard:**

```c++
#ifndef CTRL_STEP_FW_MOTION_PLANNER_H
#define CTRL_STEP_FW_MOTION_PLANNER_H
```

*   **Chinese:**  这部分代码是头文件保护符。它确保头文件只被编译一次，防止重复定义错误。`#ifndef` 检查是否已经定义了 `CTRL_STEP_FW_MOTION_PLANNER_H`。如果未定义，则执行 `#define` 定义该宏，然后编译头文件的内容。`#endif` 结束条件编译块。
*   **Explanation:** This is a standard header guard. It prevents the compiler from processing the header file multiple times in a single compilation unit.

**2. Includes:**

```c++
#include <cstdint>
```

*   **Chinese:** 包含 `cstdint` 头文件。这个头文件提供了标准整数类型的定义，例如 `int32_t` (32位有符号整数)。
*   **Explanation:** Includes the `cstdint` header, which provides definitions for fixed-width integer types like `int32_t`. This ensures consistent integer sizes across different platforms.

**3. MotionPlanner Class:**

```c++
class MotionPlanner
{
public:
    MotionPlanner() = default;

    const int32_t CONTROL_FREQUENCY = 20000;                    // Hz
    const int32_t  CONTROL_PERIOD = 1000000 / CONTROL_FREQUENCY; // uS

    // ... (nested classes and other members)

private:
    Config_t* config = nullptr;
};
```

*   **Chinese:** 定义 `MotionPlanner` 类。此类是运动规划器的主要类，包含控制频率、配置结构体以及各种追踪器类。`private` 部分包含配置指针，用于存储配置信息。
*   **Explanation:**  The `MotionPlanner` class is the core of the motion planning system. It contains constants for control frequency and period, and nested classes to manage different aspects of motion. It also has a pointer to a `Config_t` structure, which will hold configuration parameters. The `= default` after the constructor indicates that the compiler should generate the default constructor for the class.

**4. Config_t Structure:**

```c++
    struct Config_t
    {
        int32_t encoderHomeOffset;
        int32_t caliCurrent;
        int32_t ratedCurrent;
        int32_t ratedVelocity;
        int32_t ratedVelocityAcc;
        int32_t ratedCurrentAcc;
    };
```

*   **Chinese:** 定义 `Config_t` 结构体。此结构体用于存储运动规划器的配置参数，包括编码器原点偏移量、校准电流、额定电流、额定速度、额定速度加速度和额定电流加速度。
*   **Explanation:** This structure defines the configuration parameters for the motion planner. It includes settings like encoder home offset, calibration current, rated current, rated velocity, and rated accelerations. These parameters are crucial for tuning the motion control system.

**5. CurrentTracker Class:**

```c++
    class CurrentTracker
    {
    public:
        explicit CurrentTracker(MotionPlanner* _context) :
            context(_context)
        {
        }

        int32_t goCurrent = 0;

        void Init();
        void SetCurrentAcc(int32_t _currentAcc);
        void NewTask(int32_t _realCurrent);
        void CalcSoftGoal(int32_t _goalCurrent);

    private:
        MotionPlanner* context;
        int32_t currentAcc = 0;
        int32_t currentIntegral = 0;
        int32_t trackCurrent = 0;

        void CalcCurrentIntegral(int32_t _current);
    };
    CurrentTracker currentTracker = CurrentTracker(this);
```

*   **Chinese:** 定义 `CurrentTracker` 类。此类负责跟踪和控制电流。它包含目标电流 `goCurrent`，以及初始化、设置加速度、处理新任务和计算软目标的方法。私有成员包括指向 `MotionPlanner` 的指针 `context`，以及电流加速度 `currentAcc`、电流积分 `currentIntegral` 和跟踪电流 `trackCurrent`。
*   **Explanation:** The `CurrentTracker` class manages the current control loop. It includes functions for initialization, setting the current acceleration, starting a new task based on the real current, and calculating a "soft goal" for the current. The "soft goal" likely represents a target current that the system should smoothly approach. The `currentIntegral` might be used for integral control, to eliminate steady-state errors.  `CurrentTracker currentTracker = CurrentTracker(this);` creates an instance of `CurrentTracker` as a member of the `MotionPlanner` class, passing a pointer to the `MotionPlanner` object itself as the context.

**6. VelocityTracker Class:**

```c++
    class VelocityTracker
    {
    public:
        explicit VelocityTracker(MotionPlanner* _context) :
            context(_context)
        {
        }

        int32_t goVelocity = 0;

        void Init();
        void SetVelocityAcc(int32_t _velocityAcc);
        void NewTask(int32_t _realVelocity);
        void CalcSoftGoal(int32_t _goalVelocity);

    private:
        MotionPlanner* context;
        int32_t velocityAcc = 0;
        int32_t velocityIntegral = 0;
        int32_t trackVelocity = 0;

        void CalcVelocityIntegral(int32_t _velocity);
    };
    VelocityTracker velocityTracker = VelocityTracker(this);
```

*   **Chinese:** 定义 `VelocityTracker` 类。此类负责跟踪和控制速度，与 `CurrentTracker` 类似，但控制的是速度而非电流。
*   **Explanation:**  The `VelocityTracker` class mirrors the `CurrentTracker` but focuses on velocity control. It has similar functions for initialization, setting acceleration, starting new tasks based on real velocity, and calculating soft goals. The `velocityIntegral` likely serves the same purpose as `currentIntegral`, but for velocity control.  `VelocityTracker velocityTracker = VelocityTracker(this);` creates an instance of `VelocityTracker` as a member of the `MotionPlanner` class, passing a pointer to the `MotionPlanner` object itself as the context.

**7. PositionTracker Class:**

```c++
    class PositionTracker
    {
    public:
        explicit PositionTracker(MotionPlanner* _context) :
            context(_context)
        {
        }

        int32_t go_location = 0;
        int32_t go_velocity = 0;

        void Init();
        void SetVelocityAcc(int32_t value);
        void NewTask(int32_t real_location, int32_t real_speed);
        void CalcSoftGoal(int32_t _goalPosition);

    private:
        MotionPlanner* context;
        int32_t velocityUpAcc = 0;
        int32_t velocityDownAcc = 0;
        float quickVelocityDownAcc = 0;
        int32_t speedLockingBrake = 0;
        int32_t velocityIntegral = 0;
        int32_t trackVelocity = 0;
        int32_t positionIntegral = 0;
        int32_t trackPosition = 0;

        void CalcVelocityIntegral(int32_t value);
        void CalcPositionIntegral(int32_t value);
    };
    PositionTracker positionTracker = PositionTracker(this);
```

*   **Chinese:** 定义 `PositionTracker` 类。此类负责跟踪和控制位置。它包含目标位置 `go_location` 和目标速度 `go_velocity`，以及初始化、设置加速度、处理新任务和计算软目标的方法。私有成员包括不同的加速度（向上、向下、快速向下）、速度锁定刹车、速度积分、跟踪速度、位置积分和跟踪位置。
*   **Explanation:** The `PositionTracker` manages the position control loop. It includes more sophisticated acceleration controls (separate up and down accelerations, plus a "quick" deceleration) and a "speed locking brake." The `positionIntegral` is used for integral position control.  `PositionTracker positionTracker = PositionTracker(this);` creates an instance of `PositionTracker` as a member of the `MotionPlanner` class, passing a pointer to the `MotionPlanner` object itself as the context.

**8. PositionInterpolator Class:**

```c++
    class PositionInterpolator
    {
    public:
        explicit PositionInterpolator(MotionPlanner* _context) :
            context(_context)
        {
        }

        int32_t goPosition = 0;
        int32_t goVelocity = 0;

        void Init();
        void NewTask(int32_t _realPosition, int32_t _realVelocity);
        void CalcSoftGoal(int32_t _goalPosition);

    private:
        MotionPlanner* context;
        int32_t recordPosition = 0;
        int32_t recordPositionLast = 0;
        int32_t estPosition = 0;
        int32_t estPositionIntegral = 0;
        int32_t estVelocity = 0;
    };
    PositionInterpolator positionInterpolator = PositionInterpolator(this);
```

*   **Chinese:** 定义 `PositionInterpolator` 类。此类负责位置插值，用于平滑运动轨迹。它使用记录的位置、估计的位置和速度来计算目标位置。
*   **Explanation:** This class seems to be designed for smoothing the motion trajectory. It keeps track of the recorded position, the last recorded position, the estimated position, and the estimated velocity to calculate a smoother target position. This could be used to reduce jerky movements. `PositionInterpolator positionInterpolator = PositionInterpolator(this);` creates an instance of `PositionInterpolator` as a member of the `MotionPlanner` class, passing a pointer to the `MotionPlanner` object itself as the context.

**9. TrajectoryTracker Class:**

```c++
    class TrajectoryTracker
    {
    public:
        explicit TrajectoryTracker(MotionPlanner* _context) :
            context(_context)
        {
        }

        int32_t goPosition = 0;
        int32_t goVelocity = 0;

        void Init(int32_t _updateTimeout);
        void SetSlowDownVelocityAcc(int32_t value);
        void NewTask(int32_t real_location, int32_t real_speed);
        void CalcSoftGoal(int32_t _goalPosition, int32_t _goalVelocity);

    private:
        MotionPlanner* context;
        int32_t velocityDownAcc = 0;
        int32_t dynamicVelocityAcc = 0;
        int32_t updateTime = 0;
        int32_t updateTimeout = 200; // (ms) motion set-points cmd max interval
        bool overtimeFlag = false;
        int32_t recordVelocity = 0;
        int32_t recordPosition = 0;
        int32_t dynamicVelocityAccRemainder = 0;
        int32_t velocityNow = 0;
        int32_t velovityNowRemainder = 0;
        int32_t positionNow = 0;

        void CalcVelocityIntegral(int32_t value);
        void CalcPositionIntegral(int32_t value);
    };
    TrajectoryTracker trajectoryTracker = TrajectoryTracker(this);
```

*   **Chinese:** 定义 `TrajectoryTracker` 类。此类负责跟踪整个运动轨迹，包括位置和速度。它包含减速加速度、动态加速度、更新时间和超时标志。`updateTimeout` 表示运动设定点命令的最大间隔。
*   **Explanation:** The `TrajectoryTracker` manages the overall motion trajectory, including position and velocity. It includes parameters for deceleration, "dynamic" acceleration (potentially for feedforward control), an update timeout to detect stale commands, and flags to indicate whether the system is running overtime.  `TrajectoryTracker trajectoryTracker = TrajectoryTracker(this);` creates an instance of `TrajectoryTracker` as a member of the `MotionPlanner` class, passing a pointer to the `MotionPlanner` object itself as the context.

**10. AttachConfig Method:**

```c++
    void AttachConfig(Config_t* _config);

private:
    Config_t* config = nullptr;
```

*   **Chinese:** `AttachConfig` 方法用于将配置结构体指针附加到 `MotionPlanner` 对象。`config` 指针存储配置信息的地址。
*   **Explanation:** This method allows you to attach a `Config_t` structure to the `MotionPlanner` object. The `config` pointer stores the address of the configuration information, which can then be accessed by the other methods in the class.

**How the code is used (Usage Description):**

This code defines the structure for a motion planning system. To use it:

1.  **Create a `MotionPlanner` object:**  You would create an instance of the `MotionPlanner` class.
2.  **Create a `Config_t` object:** You would populate a `Config_t` structure with the specific parameters for your system (e.g., motor ratings, encoder resolution).
3.  **Call `AttachConfig`:** You would call the `AttachConfig` method to associate the `Config_t` object with the `MotionPlanner` object.
4.  **Call `Init` on each tracker:**  You would call the `Init` method on each of the tracker objects (e.g., `currentTracker.Init()`, `velocityTracker.Init()`, etc.).
5.  **Provide New Tasks:** You would then periodically call the `NewTask` methods on the tracker objects, providing the current state of the system (e.g., `currentTracker.NewTask(actualCurrent)`, `velocityTracker.NewTask(actualVelocity)`, `positionTracker.NewTask(actualPosition, actualVelocity)`).
6.  **Get Soft Goals:** Finally, you would call the `CalcSoftGoal` methods on the tracker objects to get the target values for the next control cycle (e.g., `currentTracker.CalcSoftGoal(targetCurrent)`, `velocityTracker.CalcSoftGoal(targetVelocity)`, `positionTracker.CalcSoftGoal(targetPosition)`).

**Simple Demo (Conceptual):**

```c++
#include "CTRL_STEP_FW_MOTION_PLANNER_H"
#include <iostream>

int main() {
    MotionPlanner planner;
    MotionPlanner::Config_t config;

    // Initialize configuration
    config.encoderHomeOffset = 0;
    config.caliCurrent = 100;
    config.ratedCurrent = 1000;
    config.ratedVelocity = 500;
    config.ratedVelocityAcc = 100;
    config.ratedCurrentAcc = 200;

    planner.AttachConfig(&config);

    planner.currentTracker.Init();
    planner.velocityTracker.Init();
    planner.positionTracker.Init();
    planner.trajectoryTracker.Init(200); //update timeout

    int32_t actualCurrent = 0;
    int32_t actualVelocity = 0;
    int32_t actualPosition = 0;

    int32_t targetPosition = 1000;

    for (int i = 0; i < 100; ++i) {
        // Simulate reading sensor values (in a real system, these would come from sensors)
        actualPosition += actualVelocity; // Very simplified simulation

        // Provide the new task
        planner.positionTracker.NewTask(actualPosition, actualVelocity);
        planner.trajectoryTracker.NewTask(actualPosition, actualVelocity);

        // Calculate the soft goal
        planner.positionTracker.CalcSoftGoal(targetPosition);
        planner.trajectoryTracker.CalcSoftGoal(targetPosition, 200); // Target position and velocity

        // Print for demonstration (In a real system, you would send these values to the motor controller)
        std::cout << "Iteration: " << i << ", Target Position: " << planner.positionTracker.go_location << ", Actual Position: " << actualPosition << std::endl;

        // Simulate actuator response (again, very simplified)
        actualVelocity = planner.positionTracker.go_velocity; //adjust the speed by the target
    }

    return 0;
}
```

**Important Notes:**

*   **Implementation Details:** This header file only defines the *structure* of the motion planner.  The actual *implementation* of the methods (e.g., `CalcSoftGoal`, `CalcVelocityIntegral`) is not provided in this header file and would be in a corresponding `.cpp` file.
*   **Context Pointer:**  The `context` pointer in each of the nested classes provides a way for the nested classes to access members of the `MotionPlanner` class, if needed. This can be useful for accessing configuration parameters or other shared data.
*   **Soft Goals:** The concept of "soft goals" suggests that the motion planner is designed to smoothly approach the target values, rather than abruptly changing them. This is important for reducing wear and tear on the mechanical system and for improving the overall performance of the control loop.
*   **Real-Time Considerations:**  Motion planning often needs to happen in real-time. The `CONTROL_FREQUENCY` and `CONTROL_PERIOD` constants suggest that this system is designed for a real-time environment.

This detailed explanation should provide a good understanding of the code's purpose and structure. Remember that this is just the header file, and the actual implementation would be in a separate `.cpp` file.
