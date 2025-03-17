Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Ctrl\Signal\led_base.h`

好的，我们重新来审视并改进这段C++代码。

**核心问题:**

这段代码主要负责控制LED的状态，根据电机状态和时间来控制LED的闪烁模式，例如心跳模式和指示特定状态的闪烁。 现在的问题是，代码比较紧凑，缺乏灵活性，并且状态管理较为简单。

**改进方向:**

1.  **状态管理更清晰:** 使用枚举 (enum class) 来明确定义LED的状态，而不是简单的 `bool` 变量。
2.  **闪烁模式可配置:** 提供更灵活的配置，例如闪烁频率、占空比、以及不同状态对应的闪烁模式。
3.  **避免硬编码参数:** 将一些常量（比如心跳间隔时间）定义为可配置的参数。
4.  **错误处理和安全性:**  考虑一些潜在的错误情况，例如无效的LED ID。
5.  **解耦:** 将LED控制的逻辑从 `LedBase` 中解耦出来，使用策略模式或者观察者模式，使LED控制更加灵活。

**改进后的代码:**

```cpp
#ifndef CTRL_STEP_FW_LED_BASE_H
#define CTRL_STEP_FW_LED_BASE_H

#include <cstdint>
#include "Motor/motor.h" // 假设这个头文件定义了 Motor::State_t

// 定义LED状态
enum class LedState {
    OFF,
    ON,
    BLINKING,       // 闪烁
    HEARTBEAT,      // 心跳
    ERROR           // 错误状态
};

// 定义闪烁模式结构体
struct BlinkPattern {
    uint32_t onTimeMillis;      // LED亮的时间 (毫秒)
    uint32_t offTimeMillis;     // LED灭的时间 (毫秒)
    uint8_t blinkCount;         // 闪烁次数，0表示无限闪烁
    bool repeat = false; //是否循环闪烁
};

class LedBase {
public:
    LedBase() = default;

    // 配置心跳模式
    void EnableHeartBeat(bool enable, uint32_t intervalMillis = 1000) {
        heartBeatEnable = enable;
        heartBeatInterval = intervalMillis;
    }

    // 设置闪烁模式 (目标闪烁次数)
    void SetBlink(uint8_t blink_number, uint32_t on_time, uint32_t off_time) {
        targetBlinkNum = blink_number;
        blinkPattern.onTimeMillis = on_time;
        blinkPattern.offTimeMillis = off_time;
        blinkPattern.blinkCount = blink_number; // Use blink_number for blinkCount
        blinkPattern.repeat = false;
        blinkNum = 0;
    }

    void SetBlinkPattern(const BlinkPattern& pattern) {
        blinkPattern = pattern;
        targetBlinkNum = pattern.blinkCount;
        blinkNum = 0;
    }

    // 主循环调用函数
    void Tick(uint32_t _timeElapseMillis, Motor::State_t _state);

protected:  // 允许子类访问
    // 设置LED状态 (抽象函数，由子类实现)
    virtual void SetLedState(uint8_t _id, LedState _state) = 0;

private:
    // 心跳相关
    uint32_t timerHeartBeat = 0;
    bool heartBeatEnable = false;
    uint32_t heartBeatInterval = 1000; // 心跳间隔 (毫秒)
    bool heartBeatLedOn = false;

    // 闪烁相关
    BlinkPattern blinkPattern = {100, 100, 0}; // 默认闪烁模式
    uint8_t blinkNum = 0;  // 当前闪烁次数
    uint8_t targetBlinkNum = 0; // 目标闪烁次数
    uint32_t timerBlink = 0;
    bool blinkLedOn = false;

    // 辅助函数
    void HandleHeartBeat(uint32_t _timeElapseMillis);
    void HandleBlink(uint32_t _timeElapseMillis);
};

#endif
```

```cpp
// CTRL_STEP_FW_LED_BASE.cpp (或者你的实现文件)
#include "CTRL_STEP_FW_LED_BASE.H"

void LedBase::Tick(uint32_t _timeElapseMillis, Motor::State_t _state) {
    // 根据电机状态设置 motorEnable
    motorEnable = (_state == Motor::State_t::RUNNING);  // 假设 RUNNING 是一个电机运行状态

    HandleHeartBeat(_timeElapseMillis);
    HandleBlink(_timeElapseMillis);

    //  根据状态来设置LED (示例，根据具体需求修改)
    if (motorEnable) {
        if (heartBeatEnable) {
             SetLedState(0, LedState::HEARTBEAT); //假设ID是0
        } else if (targetBlinkNum > 0) {
            SetLedState(0, LedState::BLINKING);
        } else {
            SetLedState(0, LedState::ON); // 电机运行，常亮
        }
    } else {
        SetLedState(0, LedState::OFF); // 电机停止，熄灭
    }
}

void LedBase::HandleHeartBeat(uint32_t _timeElapseMillis) {
    if (!heartBeatEnable) return;

    timerHeartBeat += _timeElapseMillis;
    if (timerHeartBeat >= heartBeatInterval) {
        timerHeartBeat = 0;
        heartBeatLedOn = !heartBeatLedOn; // 翻转LED状态
        // 心跳不需要调用 SetLedState，Tick函数已经调用了
    }
}

void LedBase::HandleBlink(uint32_t _timeElapseMillis) {
    if (targetBlinkNum == 0) return; //  没有闪烁任务

    timerBlink += _timeElapseMillis;
    if (blinkLedOn)
    {
        if (timerBlink >= blinkPattern.onTimeMillis)
        {
            timerBlink = 0;
            blinkLedOn = false;
            if(blinkPattern.onTimeMillis!=0)
            {
                blinkNum++;
            }
        }
    } else {
        if (timerBlink >= blinkPattern.offTimeMillis) {
            timerBlink = 0;
            blinkLedOn = true;
             if(blinkPattern.offTimeMillis!=0)
            {
                blinkNum++;
            }
        }
    }

    //check end
    if (blinkNum >= targetBlinkNum && targetBlinkNum != 0) {
            if(blinkPattern.repeat){
               blinkNum = 0;
            }
            else{
              targetBlinkNum = 0;
            }
        }
}
```

**代码描述 (中文):**

*   **`LedState` 枚举:** 定义了LED的可能状态：关闭 (OFF)，常亮 (ON)，闪烁 (BLINKING)，心跳 (HEARTBEAT) 和错误 (ERROR)。 使用枚举类型比简单的 `bool` 变量更清晰易懂。
*   **`BlinkPattern` 结构体:** 定义了闪烁模式的参数，包括亮的时间、灭的时间，以及闪烁次数。 这使得闪烁模式可以灵活配置。
*   **`EnableHeartBeat` 函数:**  允许启用或禁用心跳模式，并设置心跳间隔时间。  默认心跳间隔为1秒 (1000毫秒)。
*   **`SetBlink` 函数:** 设置LED进入闪烁模式。可以设定闪烁次数和闪烁周期（亮灭时间）。
*   **`Tick` 函数:**  这是主循环调用的函数，它根据电机状态、心跳状态和闪烁状态来更新LED的状态。
*   **`HandleHeartBeat` 函数:** 处理心跳逻辑。 如果心跳模式启用，它会定时翻转LED的状态。
*   **`HandleBlink` 函数:** 处理闪烁逻辑。  如果设置了闪烁模式，它会根据设定的亮灭时间和闪烁次数来控制LED的闪烁。
*   **`SetLedState` 函数:**  这是一个纯虚函数，需要在子类中实现。  子类负责实际控制LED的硬件。

**使用示例 (中文):**

假设有一个名为 `MyLedController` 的类继承自 `LedBase`:

```cpp
#include "CTRL_STEP_FW_LED_BASE.H"
#include <iostream> // 示例中使用 cout，实际项目中替换为硬件控制代码

class MyLedController : public LedBase {
public:
    void SetLedState(uint8_t _id, LedState _state) override {
        //  这里编写控制LED硬件的代码
        std::cout << "LED " << (int)_id << " State: ";
        switch (_state) {
            case LedState::OFF:
                std::cout << "OFF" << std::endl;
                break;
            case LedState::ON:
                std::cout << "ON" << std::endl;
                break;
            case LedState::BLINKING:
                std::cout << "BLINKING" << std::endl;
                break;
            case LedState::HEARTBEAT:
                std::cout << "HEARTBEAT" << std::endl;
                break;
            case LedState::ERROR:
                std::cout << "ERROR" << std::endl;
                break;
        }
    }
};

int main() {
    MyLedController ledController;

    // 启用心跳模式
    ledController.EnableHeartBeat(true, 500); // 500毫秒心跳

    // 设置闪烁模式 (闪烁3次，亮100ms，灭200ms)
    ledController.SetBlink(3, 100, 200);

    //主循环模拟
    for (int i = 0; i < 20; ++i) {
        ledController.Tick(100, Motor::State_t::RUNNING); // 模拟电机运行状态
        // 模拟电机停止
        //if(i == 10) ledController.Tick(100, Motor::State_t::STOPPED);

        //  实际项目中需要延时一段时间
       // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

   //设定循环闪烁模式
   BlinkPattern pattern;
   pattern.onTimeMillis = 50;
   pattern.offTimeMillis = 50;
   pattern.blinkCount = 0;
   pattern.repeat = true;
   ledController.SetBlinkPattern(pattern);
   for (int i = 0; i < 20; ++i) {
        ledController.Tick(100, Motor::State_t::RUNNING); // 模拟电机运行状态
    }

    return 0;
}
```

**这个示例展示了如何使用改进后的 `LedBase` 类来控制LED的行为。**
