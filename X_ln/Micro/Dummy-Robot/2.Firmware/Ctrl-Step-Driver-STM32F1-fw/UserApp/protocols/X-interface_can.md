Lan: `cpp` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\UserApp\protocols\interface_can.cpp`

好的，让我们重新开始，并尝试提供一个更智能、模块化的代码结构，并附带中文注释和示例。

**核心目标:**

*   **模块化:** 将不同的 CAN 命令处理逻辑分离到独立的函数中，提高代码可读性和可维护性。
*   **错误处理:**  添加基本的错误检查机制，例如校验接收到的数据长度。
*   **配置管理:**  将配置相关的代码组织到一个专门的结构体或类中。
*   **注释:**  添加清晰的中文注释，解释代码的功能和设计意图。
*   **示范:** 提供简洁的示例，展示如何使用这些改进的代码。

**1. 数据结构定义 (Data Structure Definitions):**

```c++
#include "common_inc.h"
#include "configurations.h"
#include <can.h>

// 假设这些变量已经在其他地方定义
extern Motor motor;
extern EncoderCalibrator encoderCalibrator;

// CAN 消息头
CAN_TxHeaderTypeDef txHeader = {
    .StdId = 0x00,
    .ExtId = 0x00,
    .IDE = CAN_ID_STD,
    .RTR = CAN_RTR_DATA,
    .DLC = 8,
    .TransmitGlobalTime = DISABLE
};

// 定义 CAN 命令处理函数指针类型
typedef void (*CanCommandHandler)(uint8_t* _data, uint32_t _len);

// 命令处理函数声明，稍后定义
void HandleEnableMotor(uint8_t* _data, uint32_t _len);
void HandleDoCalibration(uint8_t* _data, uint32_t _len);
void HandleSetCurrent(uint8_t* _data, uint32_t _len);
void HandleSetVelocity(uint8_t* _data, uint32_t _len);
void HandleSetPosition(uint8_t* _data, uint32_t _len);
void HandleSetPositionWithTime(uint8_t* _data, uint32_t _len);
void HandleSetPositionWithVelocityLimit(uint8_t* _data, uint32_t _len);
void HandleSetNodeID(uint8_t* _data, uint32_t _len);
void HandleSetCurrentLimit(uint8_t* _data, uint32_t _len);
void HandleSetVelocityLimit(uint8_t* _data, uint32_t _len);
void HandleSetAcceleration(uint8_t* _data, uint32_t _len);
void HandleApplyHomePosition(uint8_t* _data, uint32_t _len);
void HandleSetAutoEnable(uint8_t* _data, uint32_t _len);
void HandleSetDCEKp(uint8_t* _data, uint32_t _len);
void HandleSetDCEKv(uint8_t* _data, uint32_t _len);
void HandleSetDCEKi(uint8_t* _data, uint32_t _len);
void HandleSetDCEKd(uint8_t* _data, uint32_t _len);
void HandleSetStallProtect(uint8_t* _data, uint32_t _len);
void HandleGetCurrent(uint8_t* _data, uint32_t _len);
void HandleGetVelocity(uint8_t* _data, uint32_t _len);
void HandleGetPosition(uint8_t* _data, uint32_t _len);
void HandleGetOffset(uint8_t* _data, uint32_t _len);
void HandleEraseConfigs(uint8_t* _data, uint32_t _len);
void HandleReboot(uint8_t* _data, uint32_t _len);

// 命令处理函数表
CanCommandHandler commandHandlers[] = {
    nullptr, // 0x00 - 保留
    HandleEnableMotor,      // 0x01
    HandleDoCalibration,    // 0x02
    HandleSetCurrent,       // 0x03
    HandleSetVelocity,      // 0x04
    HandleSetPosition,      // 0x05
    HandleSetPositionWithTime,      // 0x06
    HandleSetPositionWithVelocityLimit, // 0x07
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // 0x08-0x0F
    HandleSetNodeID,       // 0x11
    HandleSetCurrentLimit,   // 0x12
    HandleSetVelocityLimit,  // 0x13
    HandleSetAcceleration,   // 0x14
    HandleApplyHomePosition, // 0x15
    HandleSetAutoEnable,    // 0x16
    HandleSetDCEKp, // 0x17
    HandleSetDCEKv, // 0x18
    HandleSetDCEKi, // 0x19
    HandleSetDCEKd, // 0x1A
    HandleSetStallProtect, // 0x1B
    nullptr, nullptr, nullptr, nullptr,  // 0x1C - 0x1F
    HandleGetCurrent,      // 0x21
    HandleGetVelocity,     // 0x22
    HandleGetPosition,     // 0x23
    HandleGetOffset,       // 0x24
    nullptr, nullptr, nullptr, nullptr,  // 0x25-0x2F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // 0x30-0x3F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // 0x40-0x4F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // 0x50-0x5F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // 0x60-0x6F
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // 0x70-0x7D
    HandleEraseConfigs,    // 0x7E
    HandleReboot           // 0x7F
};
```

**描述:**

*   `CAN_TxHeaderTypeDef txHeader`:  定义了 CAN 发送消息的头信息。
*   `CanCommandHandler`:  定义了一个函数指针类型，用于指向 CAN 命令处理函数。
*   `commandHandlers`:  这是一个函数指针数组，用于存储不同 CAN 命令对应的处理函数。 这种方式使得添加或修改命令处理逻辑更加容易，只需修改这个数组即可。

**2. 命令分发函数 (Command Dispatch Function):**

```c++
void OnCanCmd(uint8_t _cmd, uint8_t* _data, uint32_t _len) {
    // 检查命令ID是否有效
    if (_cmd >= sizeof(commandHandlers) / sizeof(commandHandlers[0])) {
        // 命令ID无效，不做任何处理，或者可以添加错误处理逻辑
        return;
    }

    // 获取命令处理函数
    CanCommandHandler handler = commandHandlers[_cmd];

    // 检查处理函数是否为空
    if (handler != nullptr) {
        // 调用相应的处理函数
        handler(_data, _len);
    } else {
        // 没有找到对应的处理函数，可以添加错误处理逻辑
    }
}
```

**描述:**

*   `OnCanCmd`:  这个函数接收 CAN 命令 ID (`_cmd`)，数据 (`_data`) 和数据长度 (`_len`) 作为参数。
*   它首先检查 `_cmd` 是否在有效范围内。如果 `_cmd` 超出了 `commandHandlers` 数组的索引范围，函数直接返回，不做任何处理。
*   然后，它使用 `_cmd` 作为索引，从 `commandHandlers` 数组中获取对应的处理函数。
*   如果找到有效的处理函数（即 `handler != nullptr`），则调用该函数，并将数据和数据长度传递给它。 否则，表示没有为该命令 ID 定义处理函数。

**3. 命令处理函数示例 (Example Command Handler):**

```c++
// 0x01: Enable Motor  使能电机
void HandleEnableMotor(uint8_t* _data, uint32_t _len) {
    if (_len != 4) {
        // 数据长度错误，可以添加错误处理逻辑
        return;
    }

    // 根据接收到的数据设置电机的请求模式
    motor.controller->requestMode = (*(uint32_t*) (_data) == 1) ?
                                    Motor::MODE_COMMAND_VELOCITY : Motor::MODE_STOP;
}

// 0x02: Do Calibration  执行校准
void HandleDoCalibration(uint8_t* _data, uint32_t _len) {
    // 启动编码器校准过程
    encoderCalibrator.isTriggered = true;
}

// 0x03: Set Current SetPoint  设置电流设定值
void HandleSetCurrent(uint8_t* _data, uint32_t _len) {
    if (_len != 4) {
        // 数据长度错误，可以添加错误处理逻辑
        return;
    }

    // 如果当前不是电流控制模式，则切换到电流控制模式
    if (motor.controller->modeRunning != Motor::MODE_COMMAND_CURRENT)
        motor.controller->SetCtrlMode(Motor::MODE_COMMAND_CURRENT);

    // 设置电流设定值 (单位：mA)
    motor.controller->SetCurrentSetPoint((int32_t) (*(float*) _data * 1000));
}

// 0x23: Get Position  获取位置
void HandleGetPosition(uint8_t* _data, uint32_t _len) {
    float tmpF;

    // 获取电机当前位置
    tmpF = motor.controller->GetPosition();

    // 将浮点数转换为字节数组
    auto* b = (unsigned char*) &tmpF;
    for (int i = 0; i < 4; i++) {
        _data[i] = *(b + i);
    }

    // 添加完成状态标志
    _data[4] = motor.controller->state == Motor::STATE_FINISH ? 1 : 0;

    // 设置 CAN 消息 ID
    txHeader.StdId = (boardConfig.canNodeId << 7) | 0x23;

    // 发送 CAN 消息
    CAN_Send(&txHeader, _data);
}

void HandleSetNodeID(uint8_t* _data, uint32_t _len) {
        if (_len != 4) {
        // 数据长度错误，可以添加错误处理逻辑
        return;
    }
    boardConfig.canNodeId = *(uint32_t*) (_data);
    if (_data[4])
        boardConfig.configStatus = CONFIG_COMMIT;
}

void HandleEraseConfigs(uint8_t* _data, uint32_t _len) {
    boardConfig.configStatus = CONFIG_RESTORE;
}

void HandleReboot(uint8_t* _data, uint32_t _len) {
    HAL_NVIC_SystemReset();
}

// 其他命令处理函数... (此处省略其他命令处理函数的实现)
```

**描述:**

*   **`HandleEnableMotor`**: 使能电机。根据接收到的数据，设置电机的请求模式为速度控制或停止。
*   **`HandleDoCalibration`**: 触发编码器校准过程。
*   **`HandleSetCurrent`**: 设置电流设定值。 如果当前不是电流控制模式，则切换到电流控制模式。
*   **`HandleGetPosition`**:  获取电机当前位置，并将其通过 CAN 总线发送出去。
*   **数据长度检查**:  每个处理函数都会检查接收到的数据长度是否正确。 如果数据长度不正确，则不执行任何操作，并可以添加错误处理逻辑。
*    **配置操作**:  提供了 `HandleSetNodeID`，`HandleEraseConfigs`， `HandleReboot`等配置操作函数。

**4. 示例用法 (Example Usage):**

```c++
// 假设在 CAN 接收中断中调用 OnCanCmd
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan) {
    CAN_RxHeaderTypeDef rxHeader;
    uint8_t rxData[8];

    // 获取 CAN 消息
    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK) {
        // 提取命令 ID
        uint8_t cmdId = rxHeader.StdId & 0x7F; // 假设命令ID在StdId的低7位

        // 调用命令处理函数
        OnCanCmd(cmdId, rxData, rxHeader.DLC);
    }
}
```

**描述:**

*   **`HAL_CAN_RxFifo0MsgPendingCallback`**:  这是一个假设的 CAN 接收中断回调函数。
*   在这个函数中，我们首先获取接收到的 CAN 消息，然后提取命令 ID。
*   最后，我们调用 `OnCanCmd` 函数，将命令 ID、数据和数据长度传递给它，以便进行后续处理。

**好处:**

*   **可读性:**  代码结构更清晰，每个命令的处理逻辑都集中在一个函数中。
*   **可维护性:**  修改或添加命令处理逻辑更加容易，只需修改 `commandHandlers` 数组和相应的处理函数即可。
*   **可扩展性:**  可以轻松地添加新的 CAN 命令，只需在 `commandHandlers` 数组中添加对应的处理函数即可。
*   **错误处理:**  添加了基本的错误检查机制，可以提高代码的健壮性。

**注意:**

*   这只是一个示例，你需要根据你的实际需求进行修改。
*   你需要实现所有命令处理函数的具体逻辑。
*   你需要根据你的硬件平台和 CAN 驱动程序进行相应的配置。

**下一步:**

1.  **完善命令处理函数:**  实现所有命令处理函数的具体逻辑，包括读取和写入电机控制器的状态、参数等。
2.  **添加错误处理机制:**  完善错误处理机制，例如记录错误日志、发送错误码等。
3.  **优化代码性能:**  根据实际情况，优化代码性能，例如减少内存分配、使用 DMA 传输等。
4.  **进行充分测试:**  进行充分的测试，确保代码的正确性和可靠性。

希望这些改进能够帮助你写出更智能、更易于维护的 CAN 命令处理代码!
