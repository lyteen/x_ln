Lan: `cpp` From`c/CodeCraft-2023\CodeCraft-2023\main_before_1.cpp`

Okay, let's restart and provide more focused, modular, and better explained code snippets. I'll also add Chinese descriptions and demos.

**1. Improved Angle Calculation and Normalization:**

```c++
#include <cmath>

// Constante
const double PI = 3.14159265358979323846;

/**
 * @brief 将角度归一化到 [-PI, PI] 范围内.
 *
 * @param angle 原始角度 (弧度).
 * @return double 归一化后的角度.
 *
 * 示例:
 *  double normalizedAngle = normalizeAngle(4 * PI); // normalizedAngle 将会是 0.
 */
double normalizeAngle(double angle) {
    while (angle > PI) {
        angle -= 2 * PI;
    }
    while (angle < -PI) {
        angle += 2 * PI;
    }
    return angle;
}

/**
 * @brief 计算两个角度之间的最小差值.
 *
 * @param angle1 第一个角度 (弧度).
 * @param angle2 第二个角度 (弧度).
 * @return double 最小角度差值.
 *
 * 示例:
 *  double angleDiff = angleDifference(0, PI/2); // angleDiff 将会是 PI/2.
 */
double angleDifference(double angle1, double angle2) {
    double diff = normalizeAngle(angle2 - angle1);
    return diff;
}

/**
 * @brief 计算一个向量与x轴正方向的夹角,并将其标准化到[-PI, PI]
 *
 * @param dx 向量的x分量
 * @param dy 向量的y分量
 * @return double 向量的角度 (弧度).
 *
 * 示例:
 *  double angle = vectorToAngle(1, 1); // angle 将会是 PI/4.
 */
double vectorToAngle(double dx, double dy) {
  return atan2(dy, dx);
}


// Demonstration/Example Usage:
int main() {
    double angle1 = 3.5 * PI;
    double angle2 = -2.7 * PI;

    double normalized_angle1 = normalizeAngle(angle1);
    double normalized_angle2 = normalizeAngle(angle2);

    double diff = angleDifference(normalized_angle1, normalized_angle2);

    std::cout << "Original Angle 1: " << angle1 << std::endl;
    std::cout << "Normalized Angle 1: " << normalized_angle1 << std::endl;

    std::cout << "Original Angle 2: " << angle2 << std::endl;
    std::cout << "Normalized Angle 2: " << normalized_angle2 << std::endl;

    std::cout << "Angle Difference: " << diff << std::endl;

    return 0;
}
```

**Description (中文描述):**

This code provides functions for handling angles correctly, including normalizing them to the range [-PI, PI] and calculating the shortest difference between two angles.  These functions are important for robot navigation and collision avoidance.

*   **`normalizeAngle(double angle)` (角度标准化):** 这个函数将任何角度值转换到 -PI 到 PI 的范围内。 例如，4\*PI会被转换为 0。

*   **`angleDifference(double angle1, double angle2)` (角度差值计算):** 这个函数计算两个角度之间的最小差。 它使用`normalizeAngle`来确保结果是最小差值。

*  **`vectorToAngle(double dx, double dy)` (向量转角度):**  基于向量的x和y分量，计算向量与x轴正方向的夹角.

**Demo:** The `main` function shows how to use these functions, taking some angles, normalizing them, and calculating the difference.

**2.  Improved Collision Prediction (碰撞预测):**

```c++
#include <cmath>
#include <vector>

// Constants
const double ROBOT_RADIUS = 0.5; //机器人半径
const int PREDICTION_STEPS = 10;  // 预测步数

// Structure
typedef struct robot {
    int robotId;
    double coordinate_x;
    double coordinate_y;
    double linear_velocity_x;
    double linear_velocity_y;
    double orientation;  // 弧度
} robot;

/**
 * @brief 预测在给定步数后两个机器人的距离是否会小于安全距离.
 *
 * @param robot1  第一个机器人
 * @param robot2  第二个机器人
 * @param steps   预测的步数
 * @return true  如果预测会发生碰撞, false  否则.
 *
 * 示例:
 *  if (willCollide(robot1, robot2, 5)) {
 *     //采取避碰措施
 *  }
 */
bool willCollide(const robot& robot1, const robot& robot2, int steps) {
    double x1 = robot1.coordinate_x;
    double y1 = robot1.coordinate_y;
    double x2 = robot2.coordinate_x;
    double y2 = robot2.coordinate_y;

    for (int i = 0; i <= steps; ++i) {
        x1 += robot1.linear_velocity_x / 50.0;  // 假设帧率为 50
        y1 += robot1.linear_velocity_y / 50.0;
        x2 += robot2.linear_velocity_x / 50.0;
        y2 += robot2.linear_velocity_y / 50.0;

        double distance = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));

        if (distance < 2 * ROBOT_RADIUS) {  // 假设机器人直径略大于 1
            return true;
        }
    }
    return false;
}

// Demonstration/Example Usage:
int main() {
    robot robot1 = {1, 1.0, 1.0, 1.0, 0.0, 0.0}; //id,x,y,vx,vy,angle
    robot robot2 = {2, 2.0, 1.0, -1.0, 0.0, 0.0};

    if (willCollide(robot1, robot2, PREDICTION_STEPS)) {
        std::cout << "Collision predicted!" << std::endl;
    } else {
        std::cout << "No collision predicted." << std::endl;
    }

    return 0;
}
```

**Description (中文描述):**

This code provides a more accurate collision prediction function. It simulates the movement of two robots over a number of steps and checks if their distance will be less than a safe distance.

*   **`willCollide(const robot& robot1, const robot& robot2, int steps)` (碰撞预测):**  这个函数基于给定的步数预测两个机器人是否会发生碰撞。它模拟机器人的移动，并在每一步检查距离。如果距离小于 `2 * ROBOT_RADIUS`, 则预测会发生碰撞。

**Demo:** The `main` function sets up two robot structures and calls `willCollide` to check if they will collide within `PREDICTION_STEPS`.

**3.  Modularized Robot Movement (模块化机器人运动):**

```c++
#include <iostream>
#include <cmath>

// Constants
const double MAX_SPEED = 6.0;
const double MAX_TURN_RATE = 3.14; //假设最大转向速率

// Structure
typedef struct robot {
    int robotId;
    double coordinate_x;
    double coordinate_y;
    double orientation;
    double destinationDistance_x;
    double destinationDistance_y;
} robot;

/**
 * @brief 计算到达目标点所需的旋转角度.
 *
 * @param robot  机器人
 * @param target_x  目标点的 x 坐标
 * @param target_y  目标点的 y 坐标
 * @return double  旋转角度
 */
double calculateTurnAngle(const robot& robot, double target_x, double target_y) {
    double dx = target_x - robot.coordinate_x;
    double dy = target_y - robot.coordinate_y;

    //计算目标方向与x轴的夹角
    double target_angle = atan2(dy, dx);
    //计算机器人当前朝向与目标方向的差值
    double turn_angle = target_angle - robot.orientation;

    //标准化角度
    while (turn_angle > M_PI) turn_angle -= 2 * M_PI;
    while (turn_angle < -M_PI) turn_angle += 2 * M_PI;
    return turn_angle;
}

/**
 * @brief 控制机器人朝向目标点.
 *
 * @param robot  机器人
 * @param target_x  目标点的 x 坐标
 * @param target_y  目标点的 y 坐标
 * @param turn_speed  旋转速度
 * @return void
 */
void turnTowardsTarget(robot& robot, double target_x, double target_y, double turn_speed) {
    double turn_angle = calculateTurnAngle(robot, target_x, target_y);

    if (std::abs(turn_angle) > 0.01) { //避免震荡
        double rotation = std::min(std::max(turn_angle, -turn_speed), turn_speed); //限制旋转速度
        robot.orientation += rotation; //更新机器人朝向
        std::cout << "rotate " << robot.robotId << " " << rotation << std::endl;
    }
}
/**
 * @brief 移动机器人到目标点.
 *
 * @param robot  机器人
 * @param target_x  目标点的 x 坐标
 * @param target_y  目标点的 y 坐标
 * @param speed  移动速度
 * @return void
 */
void moveToTarget(robot& robot, double target_x, double target_y, double speed) {
  // 计算距离
  double dx = target_x - robot.coordinate_x;
  double dy = target_y - robot.coordinate_y;
  double distance = sqrt(dx * dx + dy * dy);

  if (distance > 0.1) { // 如果机器人离目标点足够远
      speed = std::min(speed, distance);  // 限制速度，防止超调
      double move_x = speed * cos(robot.orientation); // 计算速度分量
      double move_y = speed * sin(robot.orientation);
      std::cout << "forward " << robot.robotId << " " << speed << std::endl; // 发送移动指令
  } else {
      std::cout << "forward " << robot.robotId << " " << 0 << std::endl; // 停止移动
  }
}

// Demonstration/Example Usage:
int main() {
    robot my_robot = {1, 0.0, 0.0, 0.0, 0.0, 0.0};
    double target_x = 10.0;
    double target_y = 5.0;

    turnTowardsTarget(my_robot, target_x, target_y, MAX_TURN_RATE / 2);  // 使用最大旋转速率的一半
    moveToTarget(my_robot, target_x, target_y, MAX_SPEED);
    return 0;
}
```

**Description (中文描述):**

This code breaks down the robot movement into smaller, more manageable functions.  This makes the code easier to understand, test, and modify.

*   **`calculateTurnAngle(const robot& robot, double target_x, double target_y)` (计算旋转角度):** 这个函数计算机器人需要旋转多少度才能朝向目标点。

*   **`turnTowardsTarget(robot& robot, double target_x, double target_y, double turn_speed)` (朝向目标):** 这个函数控制机器人旋转，使其朝向目标点。  它限制了旋转速度。

*  **`moveToTarget(robot& robot, double target_x, double target_y, double speed)` (移动到目标点):** 这个函数使机器人向目标点移动。它会根据与目标点的距离调整速度。

**Demo:** The `main` function creates a robot, sets a target, and then calls `turnTowardsTarget` and `moveToTarget` to make the robot move.

These modularized code snippets provide a foundation for more sophisticated robot control logic. By combining these components, you can implement collision avoidance, path planning, and other advanced features.
