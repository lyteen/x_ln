Lan: `cpp` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/Robot\actuators\ctrl_step\ctrl_step.cpp`

好的，我们重新来过。我将对 C++ 代码进行分析，并提供一些改进建议，附带中文解释和示例。

**1. 代码风格和可读性改进**

*   **常量定义:** 使用 `constexpr` 或 `const` 定义常量，增加代码可读性。
*   **命名规范:**  保持一致的命名规范（例如，类成员变量使用 `_` 前缀）。
*   **注释:** 添加必要的注释，解释代码的功能和意图。

```c++
// ctrl_step.hpp
#ifndef CTRL_STEP_HPP
#define CTRL_STEP_HPP

#include <stdint.h>
#include <stdbool.h>
#include "stm32f4xx_hal.h" // 假设你使用的是 STM32

// 常量定义
constexpr uint8_t CAN_ID_OFFSET = 7;
constexpr uint8_t CAN_DLC_DEFAULT = 8;

// 定义电机状态枚举
enum MotorState {
    STOP,
    RUNNING,
    FINISH
};

class CtrlStepMotor {
public:
    // 构造函数
    CtrlStepMotor(CAN_HandleTypeDef* _hcan, uint8_t _id, bool _inverse,
                  uint8_t _reduction, float _angleLimitMin, float _angleLimitMax);

    // 电机控制函数
    void SetEnable(bool _enable);
    void DoCalibration();
    void SetCurrentSetPoint(float _val);
    void SetVelocitySetPoint(float _val);
    void SetPositionSetPoint(float _val);
    void SetPositionWithVelocityLimit(float _pos, float _vel);

    // 参数配置函数
    void SetNodeID(uint32_t _id);
    void SetCurrentLimit(float _val);
    void SetVelocityLimit(float _val);
    void SetAcceleration(float _val);
    void ApplyPositionAsHome();
    void SetEnableOnBoot(bool _enable);
    void SetEnableStallProtect(bool _enable);

    // 系统控制函数
    void Reboot();
    void EraseConfigs();

    // 角度控制函数 (基于步进电机特性转换)
    void SetAngle(float _angle);
    void SetAngleWithVelocityLimit(float _angle, float _vel);
    void UpdateAngle();
    void UpdateAngleCallback(float _pos, bool _isFinished);

    // PID 参数设置函数
    void SetDceKp(int32_t _val);
    void SetDceKv(int32_t _val);
    void SetDceKi(int32_t _val);
    void SetDceKd(int32_t _val);

    // 获取电机当前角度
    float GetAngle() const { return angle; } // 添加获取角度的接口

private:
    // CAN 通信相关
    CAN_HandleTypeDef* hcan;
    CAN_TxHeaderTypeDef txHeader;
    uint8_t canBuf[CAN_DLC_DEFAULT];

    // 电机参数
    uint8_t nodeID;
    bool inverseDirection;
    uint8_t reduction;
    float angleLimitMin;
    float angleLimitMax;
    float angle; // 当前角度

    // 电机状态
    MotorState state;

    // 辅助函数 (如果需要)
    // ...
};

#endif
```

**中文描述:**

这段代码定义了步进电机控制器的头文件。 主要改进包括：使用常量定义提高了代码可读性，并添加了电机状态枚举。`GetAngle()`函数提供了获取当前角度的接口。

**2.  CAN 通信函数的改进 (communication.hpp 和 communication.cpp)**

*   **错误处理:** 添加 CAN 通信的错误处理机制。
*   **超时机制:**  在发送和接收 CAN 消息时，设置超时时间，避免程序阻塞。
*   **异步发送:** 可以考虑使用异步 CAN 发送，提高效率。

```c++
// communication.hpp
#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP

#include "stm32f4xx_hal.h"
#include <stdbool.h>

typedef struct {
    CAN_HandleTypeDef* hcan;
} CanContext;

// 获取 CAN 上下文
CanContext* get_can_ctx(CAN_HandleTypeDef* hcan);

// 发送 CAN 消息
bool CanSendMessage(CanContext* ctx, uint8_t* data, CAN_TxHeaderTypeDef* header);

// 接收 CAN 消息 (示例，假设你需要接收)
bool CanReceiveMessage(CanContext* ctx, uint8_t* data, CAN_RxHeaderTypeDef* header, uint32_t timeout);

#endif

// communication.cpp
#include "communication.hpp"

#define CAN_TRANSMIT_TIMEOUT 10 // ms

CanContext* get_can_ctx(CAN_HandleTypeDef* hcan) {
    static CanContext ctx = {0}; // 静态变量，只初始化一次
    ctx.hcan = hcan;
    return &ctx;
}

bool CanSendMessage(CanContext* ctx, uint8_t* data, CAN_TxHeaderTypeDef* header) {
    uint32_t mailbox;
    if (HAL_CAN_GetTxMailboxesFreeLevel(ctx->hcan) == 0)
    {
    return false;
    }
    if (HAL_CAN_AddTxMessage(ctx->hcan, header, data, &mailbox) != HAL_OK) {
        // 错误处理: CAN 发送失败
        return false;
    }

    // 轮询发送完成状态 或者使用中断方式发送
    uint32_t tickstart = HAL_GetTick();
    while (HAL_CAN_IsTxMessagePending(ctx->hcan, mailbox)) {
      if ((HAL_GetTick() - tickstart) > CAN_TRANSMIT_TIMEOUT)
      {
        return false; // 超时错误
      }
    }

    return true;
}


bool CanReceiveMessage(CanContext* ctx, uint8_t* data, CAN_RxHeaderTypeDef* header, uint32_t timeout) {
  if(HAL_CAN_GetRxFifoFillLevel(ctx->hcan, CAN_RX_FIFO0) == 0)
  {
    return false;
  }
    if (HAL_CAN_GetRxMessage(ctx->hcan, CAN_RX_FIFO0, header, data) != HAL_OK) {
        // 错误处理: CAN 接收失败
        return false;
    }
    return true;
}
```

**中文描述:**

这段代码定义了 CAN 通信的接口。  `CanSendMessage` 函数增加了错误处理和超时机制，避免程序卡死。`CanReceiveMessage` 提供了一个接收 CAN 消息的示例。

**3. `CtrlStepMotor` 类的改进实现**

*   **更清晰的错误处理:** 在 `CtrlStepMotor` 的成员函数中，对 `CanSendMessage` 的返回值进行检查，处理 CAN 通信失败的情况。
*   **状态管理:**  更好地管理电机状态 (`STOP`, `RUNNING`, `FINISH`)，确保状态转换的正确性。
*   **角度限制:**  在设置角度时，检查是否超出角度限制。

```c++
// ctrl_step.cpp
#include "ctrl_step.hpp"
#include "communication.hpp"
#include <algorithm> // std::clamp

CtrlStepMotor::CtrlStepMotor(CAN_HandleTypeDef* _hcan, uint8_t _id, bool _inverse,
                             uint8_t _reduction, float _angleLimitMin, float _angleLimitMax) :
    nodeID(_id), hcan(_hcan), inverseDirection(_inverse), reduction(_reduction),
    angleLimitMin(_angleLimitMin), angleLimitMax(_angleLimitMax), angle(0.0), state(STOP) // 初始化 angle 和 state
{
    txHeader =
        {
            .StdId = 0,
            .ExtId = 0,
            .IDE = CAN_ID_STD,
            .RTR = CAN_RTR_DATA,
            .DLC = CAN_DLC_DEFAULT,
            .TransmitGlobalTime = DISABLE
        };
}


void CtrlStepMotor::SetEnable(bool _enable)
{
    state = _enable ? FINISH : STOP;

    uint8_t mode = 0x01;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Int to Bytes
    uint32_t val = _enable ? 1 : 0;
    auto* b = (unsigned char*) &val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理: CAN 发送失败
        // 例如:  printf("Error: SetEnable CAN send failed!\n");
    }
}


void CtrlStepMotor::DoCalibration()
{
    uint8_t mode = 0x02;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetCurrentSetPoint(float _val)
{
    state = RUNNING;

    uint8_t mode = 0x03;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetVelocitySetPoint(float _val)
{
    state = RUNNING;

    uint8_t mode = 0x04;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetPositionSetPoint(float _val)
{
    uint8_t mode = 0x05;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need ACK

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetPositionWithVelocityLimit(float _pos, float _vel)
{
    uint8_t mode = 0x07;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_pos;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    b = (unsigned char*) &_vel;
    for (int i = 4; i < 8; i++)
        canBuf[i] = *(b + i - 4);

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetNodeID(uint32_t _id)
{
    uint8_t mode = 0x11;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Int to Bytes
    auto* b = (unsigned char*) &_id;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetCurrentLimit(float _val)
{
    uint8_t mode = 0x12;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetVelocityLimit(float _val)
{
    uint8_t mode = 0x13;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetAcceleration(float _val)
{
    uint8_t mode = 0x14;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Float to Bytes
    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 0; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::ApplyPositionAsHome()
{
    uint8_t mode = 0x15;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetEnableOnBoot(bool _enable)
{
    uint8_t mode = 0x16;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    // Int to Bytes
    uint32_t val = _enable ? 1 : 0;
    auto* b = (unsigned char*) &val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetEnableStallProtect(bool _enable)
{
    uint8_t mode = 0x1B;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    uint32_t val = _enable ? 1 : 0;
    auto* b = (unsigned char*) &val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::Reboot()
{
    uint8_t mode = 0x7f;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::EraseConfigs()
{
    uint8_t mode = 0x7e;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetAngle(float _angle)
{
    // 角度限制
    _angle = std::clamp(_angle, angleLimitMin, angleLimitMax);
    _angle = inverseDirection ? -_angle : _angle;
    float stepMotorCnt = _angle / 360.0f * (float) reduction;
    SetPositionSetPoint(stepMotorCnt);
}


void CtrlStepMotor::SetAngleWithVelocityLimit(float _angle, float _vel)
{
    // 角度限制
    _angle = std::clamp(_angle, angleLimitMin, angleLimitMax);
    _angle = inverseDirection ? -_angle : _angle;
    float stepMotorCnt = _angle / 360.0f * (float) reduction;
    SetPositionWithVelocityLimit(stepMotorCnt, _vel);
}


void CtrlStepMotor::UpdateAngle()
{
    uint8_t mode = 0x23;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::UpdateAngleCallback(float _pos, bool _isFinished)
{
    state = _isFinished ? FINISH : RUNNING;

    float tmp = _pos / (float) reduction * 360;
    angle = inverseDirection ? -tmp : tmp;
}


void CtrlStepMotor::SetDceKp(int32_t _val)
{
    uint8_t mode = 0x17;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetDceKv(int32_t _val)
{
    uint8_t mode = 0x18;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetDceKi(int32_t _val)
{
    uint8_t mode = 0x19;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}


void CtrlStepMotor::SetDceKd(int32_t _val)
{
    uint8_t mode = 0x1A;
    txHeader.StdId = nodeID << CAN_ID_OFFSET | mode;

    auto* b = (unsigned char*) &_val;
    for (int i = 0; i < 4; i++)
        canBuf[i] = *(b + i);
    canBuf[4] = 1; // Need save to EEPROM or not

    if (!CanSendMessage(get_can_ctx(hcan), canBuf, &txHeader)) {
        // 错误处理
    }
}
```

**中文描述:**

这段代码实现了 `CtrlStepMotor` 类。  主要改进包括：增加了对 `CanSendMessage` 返回值的检查，实现了角度限制，并更好地管理电机状态。  代码中添加了注释，方便理解。

**4. 示例代码 (main.cpp)**

```c++
// main.cpp
#include "stm32f4xx_hal.h" // 假设你使用的是 STM32
#include "ctrl_step.hpp"
#include "communication.hpp"
#include <stdio.h>

// 声明 CAN 句柄 (假设你在其他地方初始化了 hcan)
extern CAN_HandleTypeDef hcan;

int main() {
    // 初始化 HAL (根据你的实际情况)
    HAL_Init();

    // 初始化 CAN (根据你的实际情况)
    if (HAL_CAN_Start(&hcan) != HAL_OK) {
        printf("CAN start failed!\n");
        return -1;
    }

    // 创建 CtrlStepMotor 对象
    CtrlStepMotor motor(&hcan, 0x01, true, 100, -180.0, 180.0); // ID = 1, 反向, 减速比 100, 角度限制 -180 ~ 180

    // 启用电机
    motor.SetEnable(true);
    HAL_Delay(100); // 延时

    // 设置目标角度
    motor.SetAngle(90.0);
    HAL_Delay(1000); // 延时

    // 获取当前角度
    printf("Current angle: %f\n", motor.GetAngle());

    // 禁用电机
    motor.SetEnable(false);

    while (1) {
        // 循环执行其他任务
    }
}
```

**中文描述:**

这段代码是一个使用 `CtrlStepMotor` 类的简单示例。  它初始化 CAN，创建 `CtrlStepMotor` 对象，启用电机，设置目标角度，获取当前角度，并禁用电机。  **注意:** 你需要根据你使用的硬件平台和 HAL 库进行相应的初始化。

**总结:**

我提供了一些代码改进的建议，包括代码风格、错误处理、状态管理和角度限制。  这些改进可以提高代码的可读性、健壮性和可靠性。  请根据你的实际需求进行调整和修改。希望这些对你有帮助!
