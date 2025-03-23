Lan: `cpp` From`c/CodeCraft-2023\CodeCraft-2023\main_before_2.cpp`

好的，我将重新开始，并提供代码的关键部分解释和中文注释，以及每个小代码段的描述和简单示例。

```c++
// 机器人避障的 case 语句
// 这是机器人避障逻辑中的一部分，用于处理不同的碰撞情况。
// 具体根据两个机器人相对位置的角度，选择不同的避让策略。
switch (collisionCase) {
    case 5://四象限内半部分,角度在第三象限，自己顺时针旋转
        printf("forward %d %f\n", robot1.robotId, 1.0);  //机器人前进。
        printf("rotate %d %f\n", robot1.robotId, (-1.0) * hyperParameterRobotAngle); //机器人顺时针旋转。
        break;
    case 6://四象限内半部分,角度在第一象限,不好办,停下等对方走,让不出多少空间,还是得有一点速度.
        printf("forward %d %f\n", robot1.robotId, 1.0);  //机器人前进。
        printf("rotate %d %f\n", robot1.robotId, hyperParameterRobotAngle);   //机器人逆时针旋转
        break;
    case 7://四象限内半部分,角度在第二象限,双方都逆时针旋转
        printf("forward %d %f\n", robot1.robotId, hyperParameterAvoidSpeed); //机器人1以避让速度前进
        printf("rotate %d %f\n", robot1.robotId, hyperParameterRobotAngle);  // 机器人1 逆时针旋转
        printf("forward %d %f\n", minRobotIndex, hyperParameterAvoidSpeed); //另一个机器人以避让速度前进
        printf("rotate %d %f\n", minRobotIndex, hyperParameterRobotAngle);  // 另一个机器人逆时针旋转
        break;
    case 8://一象限内半部分,角度在第三象限,双方都逆时针旋转
        printf("forward %d %f\n", robot1.robotId, hyperParameterAvoidSpeed);  //机器人1以避让速度前进
        printf("rotate %d %f\n", robot1.robotId, (-1.0) * hyperParameterRobotAngle); // 机器人1 顺时针旋转
        printf("forward %d %f\n", minRobotIndex, hyperParameterAvoidSpeed); //另一个机器人以避让速度前进
        printf("rotate %d %f\n", minRobotIndex, (-1.0) * hyperParameterRobotAngle); // 另一个机器人顺时针旋转
        break;
    case 9://一象限内半部分,在靠近y轴的正半轴,对方减速,自己冲过去.//
        printf("forward %d %d\n", robot1.robotId, 6); //机器人1 全速前进 (这里6可能是一个速度常量)
        printf("forward %d %f\n", minRobotIndex, 1.0);   // 另一个机器人以速度1.0前进
        break;
    case 10://四象限内半部分,在靠近y轴的负半轴,对方减速,自己冲过去.//
        printf("forward %d %d\n", robot1.robotId, 6);  //机器人1 全速前进 (这里6可能是一个速度常量)
        printf("forward %d %f\n", minRobotIndex, 1.0);   // 另一个机器人以速度1.0前进
        break;
}
```

**描述:** 这一部分代码是机器人避障策略的核心，通过`switch`语句，针对不同的`collisionCase`（代表不同的碰撞情况），执行不同的运动指令。  指令包括前进(`forward`)和旋转(`rotate`)，并分别指定机器人ID和相应的速度/角度。 `hyperParameterRobotAngle` 和 `hyperParameterAvoidSpeed` 是预定义的参数，用于控制旋转角度和避让速度。

**如何使用:** `collisionCase` 是根据两个机器人的位置和角度计算出的值，用于确定最佳的避让策略。 该策略旨在避免碰撞，同时尽可能保持机器人的任务目标。

**演示:** 假设`collisionCase`等于7，那么两个机器人都会以避让速度前进，并且逆时针旋转。这是一种常见的避让方式，可以有效地避免碰撞。

```c++
// 计算机器人到目标点的距离和角度
// 这段代码计算机器人距离目标点的距离和角度，用于导航和控制机器人移动。
robot1.destinationDistance_x = worktable1.coordinate_x - robot1.coordinate_x; //计算x轴方向上的距离
robot1.destinationDistance_y = worktable1.coordinate_y - robot1.coordinate_y; //计算y轴方向上的距离

//distanceLength距离
double distanceLength = sqrt(robot1.destinationDistance_x * robot1.destinationDistance_x +
                             robot1.destinationDistance_y * robot1.destinationDistance_y);  //使用勾股定理计算直线距离
double angleOrientation_DirectionDestination;//朝向与目的地方向的夹角

if (robot1.worktable_id != worktable1.worktableArrayIndex) {  // 如果机器人还没有到达目标工作台
    double cosTheta = robot1.destinationDistance_x / distanceLength; //计算夹角的cos值
    angleOrientation_DirectionDestination = acos(cosTheta); // 通过反余弦函数计算角度
    if (robot1.destinationDistance_y < 0) //如果目标点在机器人的下方
        angleOrientation_DirectionDestination = angleOrientation_DirectionDestination * (-1.0); //调整角度为负值

```

**描述:** 这段代码首先计算机器人和目标工作台之间的X轴和Y轴距离，然后使用这些距离计算两者之间的直线距离和角度。  `acos()` 函数用于计算角度，需要根据Y轴方向进行调整。

**如何使用:** 这段代码是导航系统的一部分，用于确定机器人需要移动的方向和距离，以到达目标工作台。

**演示:** 如果 `robot1` 的坐标是 (10, 10) 并且 `worktable1` 的坐标是 (20, 20)，那么 `destinationDistance_x` 将是 10，`destinationDistance_y` 也会是 10。`distanceLength` 将是 `sqrt(10*10 + 10*10)`，约为 14.14。`angleOrientation_DirectionDestination` 将是 45 度（或 π/4 弧度）。

```c++
// 机器人到达目标工作台后的操作
// 这段代码处理机器人到达目标工作台后的买卖逻辑。
else {//到了买或卖东西
    if (robot1.product_type == 0) { // 如果机器人需要购买物品
        robot1.worktableSelectionFlag = -1;  // 释放工作台的占用标志
        worktable1.raw_material_cell_robot[0] = -1; // 释放工作台的原材料格
        printf("buy %d\n", robot1.robotId); // 执行购买操作
        fflush(stdout);
        return true;
    } else { // 如果机器人需要出售物品
        robot1.worktableSelectionFlag = -1;//处理了业务,机器人不在占用工作台  // 释放工作台的占用标志
        worktable1.raw_material_cell_robot[robot1.product_type] = -1;//卖了东西,工作台的产品格不占用 // 释放工作台的产品格
        printf("sell %d\n", robot1.robotId); // 执行出售操作
        fflush(stdout);
        return true;
    }
}
```

**描述:** 当机器人到达目标工作台后，这段代码会根据机器人的 `product_type` 来决定是执行购买还是出售操作。  在完成操作后，会释放工作台的占用标志和相应的资源格，并打印相应的操作指令。

**如何使用:**  这是机器人完成任务的关键步骤。 根据机器人的任务类型，它会与工作台进行交互，买入或卖出物品。

**演示:** 假设 `robot1.product_type` 是 0 (代表需要购买)，那么代码会执行购买操作，并释放工作台的资源。

```c++
// 计算两个工作台之间的距离
// 计算两个工作台坐标点之间的距离平方，用于路径规划和工作台选择。
double distanceSquaredBetweenWorktable(worktable &worktable1, worktable &worktable2) {//计算二点距离的平方
    double destinationDistance_x = worktable1.coordinate_x - worktable2.coordinate_x;//计算目的地距离
    double destinationDistance_y = worktable1.coordinate_y - worktable2.coordinate_y;
    //distanceLength距离
    double distanceLength = destinationDistance_x * destinationDistance_x +
                            destinationDistance_y * destinationDistance_y;
    return distanceLength;
}
```

**描述:**  此函数计算两个工作台之间的距离的平方。 距离的平方比实际距离计算更快（因为它避免了 `sqrt` 操作），并且对于比较距离通常足够。

**如何使用:** 此函数用于确定哪个工作台更接近机器人或其他工作台。  这用于路径规划、资源分配和其他决策。

**演示:** 如果 `worktable1` 的坐标是 (5, 5)，`worktable2` 的坐标是 (10, 10)，那么 `distanceSquaredBetweenWorktable` 将返回 (10-5)*(10-5) + (10-5)*(10-5) = 5