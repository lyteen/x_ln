Lan: `cpp` From`c/CodeCraft-2023\CodeCraft-2023\main_before_1.cpp`

好的，重新开始分析和解释代码，并附上中文注释和示例。

```c++
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath> // 引入cmath，使用std::fabs 和 std::atan2

using namespace std;

/**
 * 改进方向:更低的碰撞几率,更快的行路算法,更快收益的调度算法
 */
double hyperParameterAngle = 0.35;//运动中运行方向偏差的最大角度
double hyperParameterDistance = 16;//避免碰撞的距离
int hyperParameterSpeed = 6;//运动的速度
double hyperParameterAvoidSpeed = 5.5;//避免碰撞运动的速度
double hyperParameterRobotAngle = 3.14;//转弯的幅度
double hyperParameterAngleCollision = 0.75;//避免碰撞的最大方向偏差角度
double hyperParameterTimeBuy = 1.0;//买东西的时间限制
double hyperParameterTurnSpeed = 0.0;//转向运动的速度
double pi = 3.141592653;
double hyperParameterPriorityMap1 = 3.0;//放大7号工作台对物品456二缺一和一缺二的优先级
int remainingProductionTime = 500;//在什么时候就可以判断7号工作台已经快生产完成了
/**
worktable_type            工作台类型   整数        范围[1, 9]
coordinate_x coordinate_y 坐标        2个浮点 x,y 范围[0,50]即地图大小
remainingProductionTime   剩余生产时间  整数        -1：表示没有生产
（未使用）                                          0：表示生产因输出格满而阻塞。
                                                >=0：表示剩余生产帧数。
rawMaterialCellState     原材料格状态   整数       二进制位表描述，例如 48(110000)表示拥有物品 4 和 5。
productCellStatus        产品格状态    整数        0：表示无。
                                                1：表示有
worktableArrayIndex      工作台数组下标 整数       记录该工作台所在数组的下标
raw_material_cell_robot  分配时材料格的 int[8]    0代表产品材料格,int[0]范围[-1,3].int[0]=-1代表没机器人想买工作台产品,int[0]=robotId代表robotId号机器人想买工作台产品
                         状态                   [1,7]代表原材料格,int[i]范围[-1,3].int[i]=-1代表没有机器人想使用该工作台的i号原材料格,
                                               int[i]=robotId,代表robotId号机器人想使用该工作台的i号原材料格
raw_material_cell        原材料格状态   char[8]  char[i]=0代表可以放第i号物品
                                               char[i]=1代表不可以放第i号物品
*/
typedef struct worktable {
    int worktable_type;
    double coordinate_x;
    double coordinate_y;
    int remainingProductionTime;
    int rawMaterialCellState;
    int productCellStatus;
    int worktableArrayIndex;
    int raw_material_cell_robot[8];//存储产品格原材料格的状态-1可以选,状态值为i的只能被机器人i可以选,数组[0]下标为-1代表产品格没人买,i代表有机器人i买
    char raw_material_cell[8];//存储原材料格的状态0可以放,1不可以放
} worktable;
/**
worktable_id                所处工作台ID 整数       -1：表示当前没有处于任何工作台附近
(未使用)                                          [0,工作台总数-1] ：表示某工作台的下标，从 0 开始，按输入顺序定。当前机器人的所有购买、出售行为均针对该工作台进行。
product_type                携带物品类型 整数      范围[0,7]。 0 表示未携带物品。  1-7 表示对应物品。
time_value_coefficient      时间价值系数 浮点      携带物品时为[0.8, 1]的浮点数，不携带物品时为 0
(未使用)
collision_value_coefficient 碰撞价值系数 浮点      携带物品时为[0.8, 1]的浮点数，不携带物品时为 0。
(未使用)
angular_velocity            角速度     浮点       单位：弧度/秒。 正数：表示逆时针。 负数：表示顺时针。
(未使用)
linear_velocity_x           线速度     2个浮点x,y 由二维向量描述线速度，单位：米/秒
linear_velocity_y
(未使用)
orientation                 朝向       浮点      弧度，范围[-π,π]。方向示例： 0：表示右方向。 π/2：表示上方向。 -π/2：表示下方向。
coordinate_x coordinate_y   坐标       2个浮点x,y 机器人坐标
destinationDistance_x       距离       2个浮点x,y 与另一物体在x轴方向与在y轴方向的距离,该变量作用为减少代码长度
destinationDistance_y
robotId                     机器人id   整数       机器人在数组robot_array中的下标
worktableSelectionFlag      flag标记   整数       记录机器人要前往的工作台
*/
typedef struct robot {
    int worktable_id;
    int product_type;
    double time_value_coefficient;
    double collision_value_coefficient;
    double angular_velocity;
    double linear_velocity_x;
    double linear_velocity_y;
    double orientation;
    double coordinate_x;
    double coordinate_y;
    double destinationDistance_x;//与目的地坐标的距离
    double destinationDistance_y;
    int robotId;
    int worktableSelectionFlag = -1;//-1没有选择前往的工作台
} robot;

/**
 *接受地图数据,但我认为没什么用,所以没有接收地图数据
 * @return
 */
bool readUntilOK() {
    char line[1024];
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'O' && line[1] == 'K') {
            return true;
        }
        //do something

    }
    return false;
}

/**
 *接收每一帧的输入数据,接收工作台信息时,使用switch将rawMaterialCellState原材料格状态转化为更容易处理的char数组形式
 * @param frame_id          当前帧数id
 * @param money             当前金额
 * @param k                 工作台个数
 * @param worktable_array   存储k个工作台信息的数组
 * @param robot_array       存储所有机器人信息的数组
 * @param worktable_array7  在第一帧时,存储工作台类型=7的工作台的数组
 * @return 通过&,返回数组worktable_array,robot_array,worktable_array7
 */
bool
input_every_frame(int frame_id, int &money, int k, vector<worktable> &worktable_array, vector<robot> &robot_array,
                  vector<worktable> &worktable_array7) {
    worktable worktable_temp;
    for (int index_r = 0; index_r < 8; index_r++) {
        worktable_temp.raw_material_cell_robot[index_r] = -1;
    }

    robot robot_temp;
    char line[1024];
    int i = 0;
    while (fgets(line, sizeof line, stdin)) {


        if (line[0] == 'O' && line[1] == 'K') {
            return true;
        }
        //do something
        if (i == 0) {
            std::stringstream ss(line);//不知道能不能导入
            ss >> money;
            i++;
            ss.str("");
            ss.clear();
            continue;
        }
        if (i == 1) {
            std::stringstream ss(line);//不知道能不能导入
            ss >> k;
            i++;
            ss.str("");
            ss.clear();
            continue;
        }
        if (i > 1 && i <= 1 + k) {
            std::stringstream ss(line);//不知道能不能导入
            if (frame_id == 1) {//第一帧数组为空,需要插入
                ss >> worktable_temp.worktable_type >> worktable_temp.coordinate_x >> worktable_temp.coordinate_y
                   >> worktable_temp.remainingProductionTime >> worktable_temp.rawMaterialCellState
                   >> worktable_temp.productCellStatus;
                worktable_array.push_back(worktable_temp);
                worktable_temp.worktableArrayIndex = i - 2;
                if (worktable_temp.worktable_type == 7) {//在第一帧记录worktable_array7
                    worktable_array7.push_back(worktable_temp);
                }
            } else {//其他帧数组不为空,直接修改
                ss >> worktable_array[i - 2].worktable_type >> worktable_array[i - 2].coordinate_x
                   >> worktable_array[i - 2].coordinate_y
                   >> worktable_array[i - 2].remainingProductionTime >> worktable_array[i - 2].rawMaterialCellState
                   >> worktable_array[i - 2].productCellStatus;
            }
            worktable_array[i - 2].worktableArrayIndex = i - 2;
            //原材料格的记录有些问题,尝试用最笨的方法修改
            switch (worktable_array[i - 2].worktable_type) {//设置原材料格的状态
                case 1://没有原材料格
                    strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                    break;
                case 2://没有原材料格
                    strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                    break;
                case 3://没有原材料格
                    strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                    break;
                case 4://有1和2,对应rawMaterialCellState四种状态,0,2(1),4(2),6(1和2)
                    switch (worktable_array[i - 2].rawMaterialCellState) {
                        case 0:
                            strcpy(worktable_array[i - 2].raw_material_cell, "10011111");
                            break;
                        case 2:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11011111");
                            break;
                        case 4:
                            strcpy(worktable_array[i - 2].raw_material_cell, "10111111");
                            break;
                        case 6:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                            break;
                    }
                    break;
                case 5://有1和3,对应rawMaterialCellState四种状态,0,2(1),8(3),10(1和3)
                    switch (worktable_array[i - 2].rawMaterialCellState) {
                        case 0:
                            strcpy(worktable_array[i - 2].raw_material_cell, "10101111");
                            break;
                        case 2:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11101111");
                            break;
                        case 8:
                            strcpy(worktable_array[i - 2].raw_material_cell, "10111111");
                            break;
                        case 10:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                            break;
                    }
                    break;
                case 6://有2和3,对应rawMaterialCellState四种状态,0,4(2),8(3),12(2和3)
                    switch (worktable_array[i - 2].rawMaterialCellState) {
                        case 0:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11001111");
                            break;
                        case 4:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11101111");
                            break;
                        case 8:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11011111");
                            break;
                        case 12:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                            break;
                    }
                    break;
                case 7://有4,5和6,对应rawMaterialCellState八种状态,0,16(4),32(5),64(6),48(4和5),80(4和6),96(5和6),112(4,5,6)
                    strcpy(worktable_array[i - 2].raw_material_cell, "11110001");
                    switch (worktable_array[i - 2].rawMaterialCellState) {
                        case 0:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11110001");
                            break;
                        case 16:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111001");
                            break;
                        case 32:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11110101");
                            break;
                        case 48:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111101");
                            break;
                        case 64:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11110011");
                            break;
                        case 80:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111011");
                            break;
                        case 96:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11110111");
                            break;
                        case 112:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                            break;
                    }
                    break;
                case 8://有7,对应rawMaterialCellState两种状态,0,128(7)
                    switch (worktable_array[i - 2].rawMaterialCellState) {
                        case 0:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111110");
                            break;
                        case 128:
                            strcpy(worktable_array[i - 2].raw_material_cell, "11111111");
                            break;
                    }
                    break;
                case 9:
                    strcpy(worktable_array[i - 2].raw_material_cell, "10000000");
            }
            i++;
            ss.str("");
            ss.clear();
            continue;
        }
        if (i > 1 + k && i <= 5 + k) {
            std::stringstream ss(line);//不知道能不能导入
            ss >> robot_array[i - 2 - k].worktable_id >> robot_array[i - 2 - k].product_type
               >> robot_array[i - 2 - k].time_value_coefficient
               >> robot_array[i - 2 - k].collision_value_coefficient >> robot_array[i - 2 - k].angular_velocity
               >> robot_array[i - 2 - k].linear_velocity_x
               >> robot_array[i - 2 - k].linear_velocity_y >> robot_array[i - 2 - k].orientation
               >> robot_array[i - 2 - k].coordinate_x
               >> robot_array[i - 2 - k].coordinate_y;
            robot_array[i - 2 - k].robotId = i - 2 - k;
            i++;
            ss.str("");
            ss.clear();
            continue;
        }
    }
    return false;
}

/**
 *计算机器人robot1与工作台worktable1的距离的平方,之所以算平方是为了降低函数的时间复杂度(好像系统不差这点时间),
 * @param robot1 机器人
 * @param worktable1 工作台
 * @return 二者距离的平方
 */
double distanceSquaredBetweenPoint(robot &robot1, worktable &worktable1) {//计算二点距离的平方
    robot1.destinationDistance_x = worktable1.coordinate_x - robot1.coordinate_x;//计算目的地距离
    robot1.destinationDistance_y = worktable1.coordinate_y - robot1.coordinate_y;
    //distanceLength距离
    double distanceLength = robot1.destinationDistance_x * robot1.destinationDistance_x +
                            robot1.destinationDistance_y * robot1.destinationDistance_y;
    return distanceLength;
}

/**
 * 求机器人robot1离哪个机器人最近,碰撞算法中用到该函数,判断是否有机器人可能因为离得太紧而相撞
 * @param robot1 机器人
 * @param robot_array 机器人数组
 * @param minRobotIndex 记录距离robot1最近的机器人的id
 * @return 返回最小距离minDistance,通过& 返回距离robot1最近的机器人的id(下标):minRobotIndex
 */
double distanceSquaredBetweenRobot(robot &robot1, vector<robot> &robot_array, int &minRobotIndex) {//计算二点距离的平方
    int i;
    double distanceLength, minDistance = 999999.9;
    for (i = 0; i < robot_array.size(); i++) {
        if (robot1.robotId != robot_array[i].robotId) {
            robot1.destinationDistance_x = robot_array[i].coordinate_x - robot1.coordinate_x;//计算目的地距离
            robot1.destinationDistance_y = robot_array[i].coordinate_y - robot1.coordinate_y;
            distanceLength = robot1.destinationDistance_x * robot1.destinationDistance_x +
                             robot1.destinationDistance_y * robot1.destinationDistance_y;
            if (minDistance > distanceLength) {
                minDistance = distanceLength;
                minRobotIndex = robot_array[i].robotId;
            }
        }
    }
    //返回最小距离
    return minDistance;
}

/**
 * 计算两个机器人前进方向的夹角,碰撞算法中用到该函数,如果两个机器人离得太紧,就计算二者前进方向的夹角,如果双方是互相朝着对方的方向行驶,则会发生碰撞
 * @param orientation1 机器人1的朝向
 * @param orientation2 机器人2的朝向
 * @return π-二者朝向的夹角
 */
double angleRobotBetweenOrientation(double orientation1, double orientation2) {
    double angle;
    if (orientation1 >= 0)
        angle = fabs(orientation1 - 3.1416 - orientation2);
    else
        angle = fabs(orientation1 + 3.1416 - orientation2);
    if (angle > 3.1416) {
        return 3.1416 * 2 - angle;
    } else
        return angle;
}

/**
 * 坐标转换和碰撞预测
 * 将机器人2的坐标和朝向转换到机器人1的局部坐标系中，然后进行碰撞预测。
 * @param robot1 机器人1
 * @param robot2 机器人2
 * @param frameID 当前帧ID，用于调试
 * @return 0表示不碰撞，其他值表示根据碰撞情况采取的动作
 */
int coordinateTransform(robot robot1, robot robot2, int frameID) {//0不撞,1顺,2逆
    double robot2x, robot2y, orientation2;
    int flag = 0;

    // 将机器人2的坐标转换到机器人1的局部坐标系
    robot2x = (robot2.coordinate_x - robot1.coordinate_x) * cos(-1 * robot1.orientation) -
              (robot2.coordinate_y - robot1.coordinate_y) * sin(-1 * robot1.orientation);
    robot2y = (robot2.coordinate_x - robot1.coordinate_x) * sin(-1 * robot1.orientation) +
              (robot2.coordinate_y - robot1.coordinate_y) * cos(-1 * robot1.orientation);

    // 计算机器人2在机器人1坐标系下的朝向差
    orientation2 = robot2.orientation - robot1.orientation;
    if (orientation2 > pi) {
        orientation2 = orientation2 - 2 * pi;
    }
    if (orientation2 < -1 * pi) {
        orientation2 = orientation2 + 2 * pi;
    }

    // 根据机器人2在机器人1坐标系下的位置和朝向差，设置碰撞标志
    if (robot2x > -1.1 && robot2x < 4.0 && robot2y > 0 && robot2y < 3) {//第一象限的情况，处理起来比较麻烦
        if (robot2y >= 1.5 && orientation2 <= -1.0 * pi / 3.0 && orientation2 >= -1.0 * pi) {//一象限外半部分,角度在第三象限
            flag = 1;//机2顺时针
        }
        if (robot2y >= 1.5 && orientation2 >= -1.0 * pi / 3.0 && orientation2 < 0) {//一象限外半部分,角度在第四象限
            flag = 2;//机2逆时针
        }
        if (robot2x > 1.1 && robot2y < 1.5) {
            if (orientation2 <= -1.0 * pi / 3.0 && orientation2 >= -1.0 * pi ||
                orientation2 >= 5.0 / 6.0 * pi && orientation2 <= pi)//一象限内半部分,角度在第三象限
                flag = 8;
            else//一象限内半部分,角度在第四象限(因为一二象限不会发生碰撞)
                flag = 5;
        }
        if (robot2x <= 1.1 && robot2y < 1.5) {//一象限内半部分,在靠近y轴的正半轴
            flag = 9;
        }

    }
    if (robot2x > -1.1 && robot2x < 4.0 && robot2y < 0 && robot2y > -3) {//第四象限的情况，处理起来比较麻烦
        if (robot2y <= -1.5 && orientation2 <= pi / 3.0 && orientation2 > 0) {//四象限外半部分,角度在第一象限
            flag = 3;//机2顺时针
        }
        if (robot2y <= -1.5 && orientation2 > pi / 3.0 && orientation2 < pi) {//四象限外半部分,角度在第二象限
            flag = 4;//机2逆时针
        }
        if (robot2x > 1.1 && robot2y > -1.5) {
            if (orientation2 > pi / 3.0 && orientation2 < pi ||
                orientation2 < -5.0 / 6.0 * pi && orientation2 >= -1 * pi)//四象限内半部分,角度在第二象限
                flag = 7;
            else
                flag = 6;//四象限内半部分,角度在第一象限
        }
        if (robot2x <= 1.1 && robot2y > -1.5) {//四象限内半部分,在靠近y轴的负半轴
            flag = 10;
        }
    }

    int i = 0;
    if (flag != 0) {
        // 预测未来100帧，判断是否会发生碰撞
        for (i = 0; i < 100; i++) {
            // 计算小球1和小球2的当前位置
            double cur_x1 = robot1.coordinate_x + (robot1.linear_velocity_x / 50.0) * i;
            double cur_y1 = robot1.coordinate_y + (robot1.linear_velocity_y / 50.0) * i;
            double cur_x2 = robot2.coordinate_x + (robot2.linear_velocity_x / 50.0) * i;
            double cur_y2 = robot2.coordinate_y + (robot2.linear_velocity_y / 50.0) * i;

            // 计算小球1和小球2的距离
            double distance = pow(cur_x2 - cur_x1, 2) + pow(cur_y2 - cur_y1, 2);

            // 如果小球1和小球2的距离小于等于它们的直径之和，则发生了碰撞
            if (distance <= 1.2) {
                break;
            }
        }
        // 如果100帧内都不会发生碰撞，则清除碰撞标志
        if (i == 100) {
            flag = 0;
        }
    }

    //    //test
    //    if (flag != 0)
    //        fprintf(stderr, "frameID %d robot1ID %d robot2ID %d robot2x %f robot2y %f orientation2 %f flag %d\n", frameID,
    //                robot1.robotId, robot2.robotId,
    //                robot2x, robot2y, orientation2, flag);
    return flag;
}
```

**代码解释：**

*   **`coordinateTransform(robot robot1, robot robot2, int frameID)`:**
    *   **功能:**  判断机器人 `robot1` 和 `robot2` 在未来一段时间内是否会发生碰撞，并根据相对位置和朝向给出相应的应对策略。
    *   **坐标转换:**  首先，将 `robot2` 的坐标和朝向转换到 `robot1` 的局部坐标系下。 这样做可以简化碰撞判断，因为现在只需要考虑 `robot2` 相对于 `robot1` 的位置。
    *   **碰撞预测:** 然后，它会模拟未来 100 帧，计算每一步 `robot1` 和 `robot2` 的位置，并判断它们之间的距离是否小于一个阈值（1.2，可以认为是两个机器人的半径之和）。  如果距离小于阈值，则认为会发生碰撞。
    *   **碰撞标志:**  `flag` 变量用于指示碰撞情况。  它的值取决于 `robot2` 在 `robot1` 的局部坐标系中的位置和朝向，用于选择合适的避让策略。

**`coordinateTransform`函数的使用演示：**

假设我们有两个机器人 `robot1` 和 `robot2`。我们可以调用 `coordinateTransform(robot1, robot2, frameID)` 来预测它们是否会发生碰撞。

```c++
// 假设已经初始化了 robot1 和 robot2
robot robot1, robot2;
int frameID = 10; // 当前帧数

// ... 初始化 robot1 和 robot2 的坐标、朝向、速度等

int collisionFlag = coordinateTransform(robot1, robot2, frameID);

if (collisionFlag != 0) {
  // 根据 collisionFlag 的值，采取相应的避让措施
  cout << "机器人 " << robot1.robotId << " 和机器人 " << robot2.robotId << " 可能会发生碰撞，采取避让措施" << endl;
} else {
  cout << "机器人 " << robot1.robotId << " 和机器人 " << robot2.robotId << " 不会发生碰撞" << endl;
}
```

这个例子展示了如何使用 `coordinateTransform` 函数来判断两个机器人是否会发生碰撞，并根据结果采取相应的行动。 `collisionFlag` 的值会告诉你应该采取什么样的避让策略。
接下来请继续解释。
