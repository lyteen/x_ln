Lan: `cpp` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\UserApp\protocols\interface_can.cpp`

Okay, let's analyze the C code provided.  I'll break it down into smaller, understandable chunks with explanations in Chinese, focusing on functionality and usage.

```c
#include "common_inc.h"
#include "configurations.h"
#include <can.h>

// 外部变量声明
extern Motor motor;
extern EncoderCalibrator encoderCalibrator;

// CAN发送消息头定义
CAN_TxHeaderTypeDef txHeader = {
    .StdId = 0x00,       // 标准ID, 初始值为0
    .ExtId = 0x00,       // 扩展ID, 初始值为0
    .IDE = CAN_ID_STD,   // ID类型, 使用标准ID
    .RTR = CAN_RTR_DATA, // 远程帧, 使用数据帧
    .DLC = 8,            // 数据长度, 8字节
    .TransmitGlobalTime = DISABLE // 关闭全局时间戳
};
```

**描述:**

*   **包含头文件:**  包含了 `common_inc.h`, `configurations.h`, 和 `<can.h>`. 这些头文件定义了项目中常用的类型、常量、配置结构体以及CAN通信相关的函数。
*   **外部变量声明:** `extern Motor motor;` 和 `extern EncoderCalibrator encoderCalibrator;`  声明了 `motor` 和 `encoderCalibrator` 变量，它们在其他地方定义，这里只是引用。 `Motor` 结构体很可能包含了电机控制相关的所有变量(比如目标速度、位置、电流等), `EncoderCalibrator` 则用来做编码器校准.
*   **CAN发送消息头:** `txHeader` 定义了CAN发送消息的格式。  `.StdId`是标准CAN ID，`.DLC`是数据长度码，这里设置为8字节。`.IDE = CAN_ID_STD` 表示使用标准ID格式，`.RTR = CAN_RTR_DATA` 表示发送的是数据帧而不是远程请求帧。`TransmitGlobalTime = DISABLE` 关闭时间戳.

**用法:**

这段代码定义了CAN通信的基础结构。 `txHeader` 会在后续发送CAN消息时使用，作为消息的格式模板。`motor` 和 `encoderCalibrator` 这两个变量, 在 `OnCanCmd` 函数中被直接操作.

**演示:**

假设我们要发送一个CAN消息，ID为0x123，数据为`0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08`。那么在使用 `CAN_Send` 函数之前，需要设置 `txHeader.StdId = 0x123;`  然后调用 `CAN_Send(&txHeader, data);` 来发送数据。

```c
void OnCanCmd(uint8_t _cmd, uint8_t* _data, uint32_t _len) {
    float tmpF;
    int32_t tmpI;

    switch (_cmd) {
        // ... (各个case) ...
    }
}
```

**描述:**

*   **`OnCanCmd` 函数:** 这是CAN命令处理的核心函数。 当接收到CAN消息时，这个函数会被调用。
*   **参数:**
    *   `_cmd`:  接收到的CAN命令字 (Command Code),  用它来区分不同的命令类型.
    *   `_data`:  指向接收到的数据的指针。
    *   `_len`:  接收到的数据的长度。
*   **`switch` 语句:**  根据接收到的命令字 `_cmd`，进入不同的 `case` 分支执行相应的操作。
*   **临时变量:** `tmpF` 和 `tmpI` 是临时的浮点数和整数变量，用于数据类型转换。

**用法:**

这个函数是整个CAN通信处理的入口点。 所有收到的CAN命令都会经过这个函数进行解析和处理。

**演示:**

假设我们通过CAN总线接收到一个命令，`_cmd` 的值为 `0x04`，`_data` 指向的数据为 `0x00 0x00 0x80 0x3F` (单精度浮点数 1.0 的十六进制表示)。那么 `switch` 语句会进入 `case 0x04:` 分支，这个分支会将 `_data` 中的浮点数 1.0 转换为电机速度设定值，并传递给电机控制器的 `SetVelocitySetPoint` 函数。

```c
        // 0x00~0x0F No Memory CMDs (不需要保存到存储器的命令)
        case 0x01:  // Enable Motor (使能电机)
            motor.controller->requestMode = (*(uint32_t*) (RxData) == 1) ?
                                            Motor::MODE_COMMAND_VELOCITY : Motor::MODE_STOP;
            break;
        case 0x02:  // Do Calibration (执行校准)
            encoderCalibrator.isTriggered = true;
            break;
        case 0x03:  // Set Current SetPoint (设置电流设定值)
            if (motor.controller->modeRunning != Motor::MODE_COMMAND_CURRENT)
                motor.controller->SetCtrlMode(Motor::MODE_COMMAND_CURRENT);
            motor.controller->SetCurrentSetPoint((int32_t) (*(float*) RxData * 1000));
            break;
        case 0x04:  // Set Velocity SetPoint (设置速度设定值)
            if (motor.controller->modeRunning != Motor::MODE_COMMAND_VELOCITY) {
                motor.config.motionParams.ratedVelocity = boardConfig.velocityLimit;
                motor.controller->SetCtrlMode(Motor::MODE_COMMAND_VELOCITY);
            }
            motor.controller->SetVelocitySetPoint(
                (int32_t) (*(float*) RxData *
                           (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS));
            break;
```

**描述:**

*   **无存储命令 (0x00 ~ 0x0F):**  这部分 `case` 处理不需要将配置保存到EEPROM或其他非易失性存储器的命令。
*   **`case 0x01`: 使能电机:**  如果接收到的 `RxData` 的前4个字节被解释为整数后等于1，则将电机的控制模式设置为 `Motor::MODE_COMMAND_VELOCITY`（速度控制模式），否则设置为 `Motor::MODE_STOP`（停止模式）。
*   **`case 0x02`: 执行校准:**  设置 `encoderCalibrator.isTriggered = true;` 触发编码器校准。
*   **`case 0x03`: 设置电流设定值:**  如果当前电机控制模式不是电流控制，则先切换到电流控制模式。然后将接收到的 `RxData` 中的浮点数乘以1000转换为整数，设置为电流设定值。乘以1000可能是因为电流的单位是毫安。
*   **`case 0x04`: 设置速度设定值:**  如果当前电机控制模式不是速度控制，则先切换到速度控制模式，并将电机的额定速度设置为板子的速度限制。 然后将接收到的 `RxData` 中的浮点数乘以 `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 转换为整数，设置为速度设定值。  `MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 表示电机转一圈对应的步数。

**用法:**

这些 `case` 实现了对电机的一些基本控制，如使能、校准、设置电流和速度。

**演示:**

*   发送命令 `0x01`，数据为 `0x01 0x00 0x00 0x00`，可以使能电机并进入速度控制模式。
*   发送命令 `0x03`，数据为 `0x00 0x00 0x80 0x3F`，可以将电机的电流设定值设置为 1.0 * 1000 = 1000 mA。
*   发送命令 `0x04`，数据为 `0x00 0x00 0x80 0x3F`，可以将电机速度设定值设置为 1.0 * `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 步/秒。

```c
        case 0x05:  // Set Position SetPoint (设置位置设定值)
            if (motor.controller->modeRunning != Motor::MODE_COMMAND_POSITION) {
                motor.config.motionParams.ratedVelocity = boardConfig.velocityLimit;
                motor.controller->SetCtrlMode(Motor::MODE_COMMAND_POSITION);
            }
            motor.controller->SetPositionSetPoint(
                (int32_t) (*(float*) RxData * (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS));
            if (_data[4]) // Need Position & Finished ACK (需要位置和完成确认)
            {
                tmpF = motor.controller->GetPosition();
                auto* b = (unsigned char*) &tmpF;
                for (int i = 0; i < 4; i++)
                    _data[i] = *(b + i);
                _data[4] = motor.controller->state == Motor::STATE_FINISH ? 1 : 0;
                txHeader.StdId = (boardConfig.canNodeId << 7) | 0x23;
                CAN_Send(&txHeader, _data);
            }
            break;
```

**描述:**

*   **`case 0x05`: 设置位置设定值:**
    *   如果当前电机控制模式不是位置控制，则先切换到位置控制模式，并将电机的额定速度设置为板子的速度限制。
    *   将接收到的 `RxData` 中的浮点数乘以 `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 转换为整数，设置为位置设定值。
    *   **位置和完成确认:** 如果 `_data[4]` 的值为真（非零），则表示需要发送位置和完成状态的确认消息。
        *   读取当前电机的位置，并将位置的浮点数值转换为字节数组，复制到 `_data` 的前4个字节。
        *   将 `_data[4]` 设置为电机的完成状态：如果电机状态为 `Motor::STATE_FINISH`，则设置为 1，否则设置为 0。
        *   构造CAN消息的ID `(boardConfig.canNodeId << 7) | 0x23`，然后通过 `CAN_Send` 函数发送确认消息。  `boardConfig.canNodeId << 7` 意味着将节点ID左移7位，然后和`0x23`或运算，得到最终的CAN ID.

**用法:**

此 `case` 用于设置电机的目标位置，并且可以选择性地请求位置和完成状态的确认消息。

**演示:**

*   发送命令 `0x05`，数据为 `0x00 0x00 0x00 0x40 0x01`，可以将电机的目标位置设置为 2.0 * `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 步，并且请求发送位置和完成状态的确认消息。收到这个命令后，会把当前的位置和运动完成标志一起发送回去.
*   发送命令 `0x05`，数据为 `0x00 0x00 0x00 0x40 0x00`，只设置电机的目标位置，不请求确认消息。

```c
        case 0x06:  // Set Position with Time (设置带时间的位置设定值)
            if (motor.controller->modeRunning != Motor::MODE_COMMAND_POSITION)
                motor.controller->SetCtrlMode(Motor::MODE_COMMAND_POSITION);
            motor.controller->SetPositionSetPointWithTime(
                (int32_t) (*(float*) RxData * (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS),
                *(float*) (RxData + 4));
            if (_data[4]) // Need Position & Finished ACK (需要位置和完成确认)
            {
                tmpF = motor.controller->GetPosition();
                auto* b = (unsigned char*) &tmpF;
                for (int i = 0; i < 4; i++)
                    _data[i] = *(b + i);
                _data[4] = motor.controller->state == Motor::STATE_FINISH ? 1 : 0;
                txHeader.StdId = (boardConfig.canNodeId << 7) | 0x23;
                CAN_Send(&txHeader, _data);
            }
            break;
```

**描述:**

*   **`case 0x06`: 设置带时间的位置设定值:**
    *   如果当前电机控制模式不是位置控制，则先切换到位置控制模式。
    *   调用 `motor.controller->SetPositionSetPointWithTime` 函数，设置目标位置和运动时间。目标位置从 `RxData` 的前4个字节读取，运动时间从 `RxData` 的第5到第8个字节读取。
    *   **位置和完成确认:**  和 `case 0x05` 类似，如果 `_data[4]` 的值为真，则发送位置和完成状态的确认消息。

**用法:**

这个 `case` 允许设置目标位置，并且指定运动完成的时间。这可以用于规划运动轨迹。

**演示:**

*   发送命令 `0x06`，数据为 `0x00 0x00 0x00 0x40 0x00 0x00 0x80 0x3F 0x01`，可以将电机的目标位置设置为 2.0 * `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 步，运动时间设置为 1.0 秒，并且请求发送位置和完成状态的确认消息。注意`0x00 0x00 0x80 0x3F`代表浮点数1.0.
*  发送命令 `0x06`，数据为 `0x00 0x00 0x00 0x40 0x00 0x00 0x80 0x3F 0x00`，只设置目标位置和运动时间, 不请求确认消息.

```c
        case 0x07:  // Set Position with Velocity-Limit (设置带速度限制的位置设定值)
        {
            if (motor.controller->modeRunning != Motor::MODE_COMMAND_POSITION)
            {
                motor.config.motionParams.ratedVelocity = boardConfig.velocityLimit;
                motor.controller->SetCtrlMode(Motor::MODE_COMMAND_POSITION);
            }
            motor.config.motionParams.ratedVelocity =
                (int32_t) (*(float*) (RxData + 4) * (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS);
            motor.controller->SetPositionSetPoint(
                (int32_t) (*(float*) RxData * (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS));
            // Always Need Position & Finished ACK (总是需要位置和完成确认)
            tmpF = motor.controller->GetPosition();
            auto* b = (unsigned char*) &tmpF;
            for (int i = 0; i < 4; i++)
                _data[i] = *(b + i);
            _data[4] = motor.controller->state == Motor::STATE_FINISH ? 1 : 0;
            txHeader.StdId = (boardConfig.canNodeId << 7) | 0x23;
            CAN_Send(&txHeader, _data);
        }
        break;
```

**描述:**

*   **`case 0x07`: 设置带速度限制的位置设定值:**
    *   如果当前电机控制模式不是位置控制，则先切换到位置控制模式，并将电机的额定速度设置为板子的速度限制。
    *   从 `RxData` 的第5到第8个字节读取速度限制值，并将其设置为电机的额定速度 `motor.config.motionParams.ratedVelocity`。
    *   从 `RxData` 的前4个字节读取目标位置，并调用 `motor.controller->SetPositionSetPoint` 函数设置目标位置。
    *   **总是需要位置和完成确认:** 这个 `case` 强制发送位置和完成状态的确认消息，无论 `_data[4]` 的值是什么。

**用法:**

这个 `case` 允许设置目标位置，并同时设置运动过程中的速度限制。 强制发送确认消息可以确保上位机知道电机已经收到了命令并开始执行。

**演示:**

*   发送命令 `0x07`，数据为 `0x00 0x00 0x00 0x40 0x00 0x00 0x80 0x3F`，可以将电机的目标位置设置为 2.0 * `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 步，速度限制设置为 1.0 * `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 步/秒。  注意，这里无论发送的数据的第5个byte是什么, 都会发送位置和完成状态的确认消息。

```c
        // 0x10~0x1F CMDs with Memory (需要保存到存储器的命令)
        case 0x11:  // Set Node-ID and Store to EEPROM (设置节点ID并保存到EEPROM)
            boardConfig.canNodeId = *(uint32_t*) (RxData);
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x12:  // Set Current-Limit and Store to EEPROM (设置电流限制并保存到EEPROM)
            motor.config.motionParams.ratedCurrent = (int32_t) (*(float*) RxData * 1000);
            boardConfig.currentLimit = motor.config.motionParams.ratedCurrent;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x13:  // Set Velocity-Limit and Store to EEPROM (设置速度限制并保存到EEPROM)
            motor.config.motionParams.ratedVelocity =
                (int32_t) (*(float*) RxData *
                           (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS);
            boardConfig.velocityLimit = motor.config.motionParams.ratedVelocity;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
```

**描述:**

*   **带存储命令 (0x10 ~ 0x1F):**  这部分 `case` 处理需要将配置保存到EEPROM或其他非易失性存储器的命令。
*   **`case 0x11`: 设置节点ID并保存到EEPROM:**  从 `RxData` 读取节点ID，并更新 `boardConfig.canNodeId`。如果 `_data[4]` 为真，则设置 `boardConfig.configStatus = CONFIG_COMMIT;`，表示需要将配置写入EEPROM。
*   **`case 0x12`: 设置电流限制并保存到EEPROM:**  从 `RxData` 读取电流限制值，并更新 `motor.config.motionParams.ratedCurrent` 和 `boardConfig.currentLimit`。 如果 `_data[4]` 为真，则设置 `boardConfig.configStatus = CONFIG_COMMIT;`。
*   **`case 0x13`: 设置速度限制并保存到EEPROM:**  从 `RxData` 读取速度限制值，并更新 `motor.config.motionParams.ratedVelocity` 和 `boardConfig.velocityLimit`。如果 `_data[4]` 为真，则设置 `boardConfig.configStatus = CONFIG_COMMIT;`。

**用法:**

这些 `case` 用于配置电机的参数，如节点ID、电流限制和速度限制，并将这些配置保存到EEPROM，以便下次启动时可以恢复。

**演示:**

*   发送命令 `0x11`，数据为 `0x05 0x00 0x00 0x00 0x01`，可以将节点ID设置为 5，并且请求将配置写入EEPROM。
*   发送命令 `0x12`，数据为 `0x00 0x00 0x80 0x42 0x01`，可以将电流限制设置为 64.0 * 1000 = 64000 mA，并且请求将配置写入EEPROM.  注意`0x00 0x00 0x80 0x42`代表浮点数64.0.
*   在实际应用中，需要一个单独的函数来检测 `boardConfig.configStatus` 的值是否为 `CONFIG_COMMIT`，如果是，则将 `boardConfig` 中的数据写入EEPROM。

```c
        case 0x14:  // Set Acceleration （and Store to EEPROM）（设置加速度并保存到EEPROM）
            tmpF = *(float*) RxData * (float) motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS;

            motor.config.motionParams.ratedVelocityAcc = (int32_t) tmpF;
            motor.motionPlanner.velocityTracker.SetVelocityAcc((int32_t) tmpF);
            motor.motionPlanner.positionTracker.SetVelocityAcc((int32_t) tmpF);
            boardConfig.velocityAcc = motor.config.motionParams.ratedVelocityAcc;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x15:  // Apply Home-Position and Store to EEPROM (应用Home-Position并保存到EEPROM)
            motor.controller->ApplyPosAsHomeOffset();
            boardConfig.encoderHomeOffset = motor.config.motionParams.encoderHomeOffset %
                                            motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS;
            boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x16:  // Set Auto-Enable and Store to EEPROM (设置自动使能并保存到EEPROM)
            boardConfig.enableMotorOnBoot = (*(uint32_t*) (RxData) == 1);
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
```

**描述:**

*   **`case 0x14`: 设置加速度并保存到EEPROM:** 从 `RxData` 读取加速度值, 乘以 `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 转换为整数, 设置 `motor.config.motionParams.ratedVelocityAcc`, 同时更新 `motor.motionPlanner.velocityTracker` 和 `motor.motionPlanner.positionTracker` 的加速度, 最后更新 `boardConfig.velocityAcc`. 如果 `_data[4]` 为真, 则设置 `boardConfig.configStatus = CONFIG_COMMIT;`.
*   **`case 0x15`: 应用 Home-Position 并保存到EEPROM:** 调用 `motor.controller->ApplyPosAsHomeOffset()` 将当前位置设置为 Home-Position (零点偏移),  将编码器的 Home Offset 保存在 `boardConfig.encoderHomeOffset` 中,  对 `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS` 取模是为了保证offset在一个圈内, 设置 `boardConfig.configStatus = CONFIG_COMMIT;`
*   **`case 0x16`: 设置自动使能并保存到EEPROM:** 从 `RxData` 读取一个整数, 如果等于1, 设置 `boardConfig.enableMotorOnBoot = true`, 否则设置为 false.  如果 `_data[4]` 为真, 则设置 `boardConfig.configStatus = CONFIG_COMMIT;`.

**用法:**

*   `0x14`: 配置电机的加速度, 同时更新运动规划器.
*   `0x15`: 应用当前位置为零点,  在下次启动时, 电机将以当前位置为零点.
*   `0x16`: 设置电机是否在启动时自动使能.

**演示:**

*   发送命令 `0x14`, 数据为 `0x00 0x00 0x80 0x40 0x01`, 可以将加速度设置为 2.0 * `motor.MOTOR_ONE_CIRCLE_SUBDIVIDE_STEPS`, 并且请求将配置写入EEPROM.
*   移动电机到一个期望的 Home 位置, 然后发送命令 `0x15`, 数据为 `0x00 0x00 0x00 0x00 0x01`,  请求将配置写入EEPROM.
*   发送命令 `0x16`, 数据为 `0x01 0x00 0x00 0x00 0x01`,  设置开机自动使能, 并且请求将配置写入EEPROM.

```c
        case 0x17:  // Set DCE Kp
            motor.config.ctrlParams.dce.kp = *(int32_t*) (RxData);
            boardConfig.dce_kp = motor.config.ctrlParams.dce.kp;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x18:  // Set DCE Kv
            motor.config.ctrlParams.dce.kv = *(int32_t*) (RxData);
            boardConfig.dce_kv = motor.config.ctrlParams.dce.kv;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x19:  // Set DCE Ki
            motor.config.ctrlParams.dce.ki = *(int32_t*) (RxData);
            boardConfig.dce_ki = motor.config.ctrlParams.dce.ki;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x1A:  // Set DCE Kd
            motor.config.ctrlParams.dce.kd = *(int32_t*) (RxData);
            boardConfig.dce_kd = motor.config.ctrlParams.dce.kd;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
        case 0x1B:  // Set Enable Stall-Protect
            motor.config.ctrlParams.stallProtectSwitch = (*(uint32_t*) (RxData) == 1);
            boardConfig.enableStallProtect = motor.config.ctrlParams.stallProtectSwitch;
            if (_data[4])
                boardConfig.configStatus = CONFIG_COMMIT;
            break;
```

**描述:**

*   **`case 0x17 - 0x1A`: 设置 DCE Kp, Kv, Ki, Kd:** 分别从 `RxData` 中读取Direct Current Error (DCE) 控制器的 Kp (比例增益), Kv (速度前馈增益), Ki (积分增益), Kd (微分增益), 更新 `motor.config.ctrlParams.dce.kp`, `motor.config.ctrlParams.dce.kv`,  `motor.config.ctrlParams.dce.ki`, `motor.config.ctrlParams.dce.kd`, 同时更新 `boardConfig.dce_kp`, `boardConfig.dce_kv`, `boardConfig.dce_ki`, `boardConfig.dce_kd`. 如果 `_data[4]` 为真, 则设置 `boardConfig.configStatus = CONFIG_COMMIT;`.
*   **`case 0x1B`: 设置使能 Stall-Protect:** 从 `RxData` 读取一个整数, 如果等于1, 设置 `motor.config.ctrlParams.stallProtectSwitch = true`, 否则设置为 false. 同时更新 `boardConfig.enableStallProtect`.  如果 `_data[4]` 为真, 则设置 `boardConfig.configStatus = CONFIG_COMMIT;`.

**用法:**

*   `0x17 - 0x1A`: 用于调整 DCE 控制器的参数, 用于优化电机控制的性能.
*   `0x1B`: 用于使能或禁用失速保护,  当电机堵转时,  失速保护会停止电机运行, 防止损坏.

**演示:**

*   发送命令 `0x17`, 数据为 `0x00 0x00 0x80 0x3F 0x01`,  可以将 DCE Kp 设置为 1.0, 并且请求将配置写入EEPROM.  注意`0x00 0x00 0x80 0x3F`代表浮点数1.0， 但这里会被强制转换成int32_t， 所以实际的值为1.
*   发送命令 `0x1B`, 数据为 `0x01 0x00 0x00 0x00 0x01`,  设置使能失速保护, 并且请求将配置写入EEPROM.

```c
        // 0x20~0x2F Inquiry CMDs (查询命令)
        case 0x21: // Get Current (获取电流)
        {
            tmpF = motor.controller->GetFocCurrent();
            auto* b = (unsigned char*) &tmpF;
            for (int i = 0; i < 4; i++)
                _data[i] = *(b + i);
            _data[4] = (motor.controller->state == Motor::STATE_FINISH ? 1 : 0);

            txHeader.StdId = (boardConfig.canNodeId << 7) | 0x21;
            CAN_Send(&txHeader, _data);
        }
        break;
        case 0x22: // Get Velocity (获取速度)
        {
            tmpF = motor.controller->GetVelocity();
            auto* b = (unsigned char*) &tmpF;
            for (int i = 0; i < 4; i++)
                _data[i] = *(b + i);
            _data[4] = (motor.controller->state == Motor::STATE_FINISH ? 1 : 0);

            txHeader.StdId = (boardConfig.canNodeId << 7) | 0x22;
            CAN_Send(&txHeader, _data);
        }
        break;
        case 0x23: // Get Position (获取位置)
        {
            tmpF = motor.controller->GetPosition();
            auto* b = (unsigned char*) &tmpF;
            for (int i = 0; i < 4; i++)
                _data[i] = *(b + i);
            // Finished ACK
            _data[4] = motor.controller->state == Motor::STATE_FINISH ? 1 : 0;
            txHeader.StdId = (boardConfig.canNodeId << 7) | 0x23;
            CAN_Send(&txHeader, _data);
        }
        break;
        case 0x24: // Get Offset (获取偏移量)
        {
            tmpI = motor.config.motionParams.encoderHomeOffset;
            auto* b = (unsigned char*) &tmpI;
            for (int i = 0; i < 4; i++)
                _data[i] = *(b + i);
            txHeader.StdId = (boardConfig.canNodeId << 7) | 0x24;
            CAN_Send(&txHeader, _data);
        }
        break;
```

**描述:**

*   **查询命令 (0x20 ~ 0x2F):**  这部分 `case` 用于查询电机当前的状态信息.
*   **`case 0x21`: 获取电流:** 调用 `motor.controller->GetFocCurrent()` 获取FOC (Field-Oriented Control, 磁场定向控制) 电流值, 将浮点数转换为字节数组, 复制到 `_data` 的前4个字节.  `_data[4]` 设置为电机的完成状态.  构造CAN消息的ID `(boardConfig.canNodeId << 7) | 0x21`, 然后发送数据.
*   **`case 0x22`: 获取速度:**  和 `0x21` 类似, 调用 `motor.controller->GetVelocity()` 获取速度值, 并发送.
*   **`case 0x23`: 获取位置:**  和 `0x21` 类似, 调用 `motor.controller->GetPosition()` 获取位置值, 并发送.
*   **`case 0x24`: 获取偏移量:**  读取 `motor.config.motionParams.encoderHomeOffset` 获取编码器偏移量, 并发送.

**用法:**

通过这些命令，上位机可以获取电机当前的运行状态，用于监控和调试。

**演示:**

*   发送命令 `0x21`, 接收到的 `_data` 的前4个字节将会是当前电机的 FOC 电流值.
*   发送命令 `0x22`, 接收到的 `_data` 的前4个字节将会是当前电机的速度值.
*   发送命令 `0x23`, 接收到的 `_data` 的前4个字节将会是当前电机的位置值.
*   发送命令 `0x24`, 接收到的 `_data` 的前4个字节将会是当前电机的编码器偏移量.

```c
        case 0x7e:  // Erase Configs (擦除配置)
            boardConfig.configStatus = CONFIG_RESTORE;
            break;
        case 0x7f:  // Reboot (重启)
            HAL_NVIC_SystemReset();
            break;
        default:
            break;
```

**描述:**

*   **`case 0x7e`: 擦除配置:** 设置 `boardConfig.configStatus = CONFIG_RESTORE;`，表示需要将配置恢复到默认值。这通常会在下次启动时执行。
*   **`case 0x7f`: 重启:** 调用 `HAL_NVIC_SystemReset()` 函数，重启系统。
*   **`default`:** 如果接收到的命令字不在任何已定义的 `case` 中，则执行 `default` 分支，通常是空操作 (`break;`)。

**用法:**

*   `0x7e`: 用于将配置恢复到出厂设置.
*   `0x7f`: 用于重启系统.

**演示:**

*   发送命令 `0x7e`, 会在下次启动时将配置恢复到出厂设置.  通常会配合单独的EEPROM写入函数来完成此操作。
*   发送命令 `0x7f`,  系统会立即重启.

**总结:**

这段C代码实现了一个基于CAN总线的电机控制接口。它定义了各种CAN命令，用于控制电机的运行模式、设置参数以及查询状态。 代码结构清晰，易于理解和扩展. 需要注意的是, 实际的EEPROM读写操作, 电机控制, 编码器校准等功能的代码并没有包含在这段代码中, 需要在其他地方实现.
