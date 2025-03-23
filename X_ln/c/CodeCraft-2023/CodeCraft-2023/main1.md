Lan: `cpp` From`c/CodeCraft-2023\CodeCraft-2023\main1.cpp`

好的，现在我将对提供的C++代码进行解释，并提供代码片段和中文描述，以及每个片段的使用说明和简单示例。

**1. `findShortestBuyDistance` 函数**

```c++
/**
 * 计算最短距离和,函数用于计算优先级
 * @param worktableIndexBuyArray 工作台id数组,存储在当前帧,能被机器人robotId所选择并且成品格有物品i的所有工作台的id.(该工作台的产品格有物品i,并且没有其他机器人标记该产品格)
 * @param worktableIndexSellArray 工作台id数组,存储在当前帧,能收购物品i的所有工作台的id(物品对应的该工作台的原材料格是空的,并且没有其他机器人标记该原材料格)
 * @param worktable_array 工作台数组,存储所有工作台信息
 * @param robot_array 机器人数组,存储所有机器人信息
 * @param robotId 机器人id
 * @param minDistance 存储最短距离和,和为机器人到工作台id=minWorktableIndex的距离平方+工作台id=minWorktableIndex的距离到工作台id=minWorktableSellIndex的距离平方
 * @param minWorktableIndex 买该物品的工作台id
 * @param minWorktableSellIndex 卖该物品的工作台id
 * 通过& 返回距离和minDistance,买物品的工作台id=minWorktableIndex,卖物品的工作台id=minWorktableSellIndex
 */
void findShortestBuyDistance(vector<int> &worktableIndexBuyArray, vector<int> &worktableIndexSellArray,
                        vector<worktable> &worktable_array, vector<robot> &robot_array,
                        int &robotId, double &minDistance, int &minWorktableIndex,
                        int &minWorktableSellIndex) {//寻找最近的买卖距离和
    double distanceBetween, distanceBetweenPoint;
    minDistance = 9999999.9;
    int i, j;
    for (i = 0; i < worktableIndexBuyArray.size(); i++) {
        distanceBetweenPoint = distanceSquaredBetweenPoint(robot_array[robotId],
                                                           worktable_array[worktableIndexBuyArray[i]]);
        for (j = 0; j < worktableIndexSellArray.size(); j++) {
            distanceBetween = distanceBetweenPoint
                              +
                              distanceSquaredBetweenWorktable(worktable_array[worktableIndexBuyArray[i]],
                                                              worktable_array[worktableIndexSellArray[j]]);
            if (minDistance > distanceBetween) {//如果是最短路径,记录距离和走过的两个结点
                minDistance = distanceBetween;
                minWorktableSellIndex = worktableIndexSellArray[j];
                minWorktableIndex = worktableIndexBuyArray[i];
            }
        }
    }
}
```

**描述:**

这个函数的目标是找到一个机器人 `robotId` 买卖物品的“最佳”工作台组合，这里的“最佳”定义为机器人到购买工作台的距离 + 购买工作台到出售工作台的距离之和最小。

**使用说明:**

1.  **输入:**
    *   `worktableIndexBuyArray`:  存储了所有可以购买物品的工作台的 ID 列表。
    *   `worktableIndexSellArray`: 存储了所有可以出售物品的工作台的 ID 列表。
    *   `worktable_array`: 存储了所有工作台信息的数组。
    *   `robot_array`: 存储了所有机器人信息的数组。
    *   `robotId`:  指定要进行买卖操作的机器人的 ID。
    *   `minDistance`, `minWorktableIndex`, `minWorktableSellIndex`: 用于存储结果的变量，`minDistance` 存储最小距离和，`minWorktableIndex` 存储购买工作台的 ID，`minWorktableSellIndex` 存储出售工作台的 ID。

2.  **流程:**
    *   初始化 `minDistance` 为一个非常大的值。
    *   遍历所有可能的购买工作台（`worktableIndexBuyArray`）。
    *   对于每个购买工作台，遍历所有可能的出售工作台（`worktableIndexSellArray`）。
    *   计算机器人到购买工作台的距离 `distanceBetweenPoint`。
    *   计算购买工作台到出售工作台的距离 `distanceBetween`。
    *   如果 `distanceBetween` 小于当前的 `minDistance`，则更新 `minDistance`，`minWorktableIndex` 和 `minWorktableSellIndex`。

3.  **输出:**
    *   通过引用传递，`minDistance`，`minWorktableIndex` 和 `minWorktableSellIndex` 将被更新为找到的最佳值。

**简单示例:**

假设你有以下数据:

```c++
vector<worktable> worktable_array = {
    {10.0, 10.0}, // 工作台 0
    {20.0, 20.0}, // 工作台 1
    {30.0, 30.0}  // 工作台 2
};

vector<robot> robot_array = {
    {5.0, 5.0} // 机器人 0
};

vector<int> worktableIndexBuyArray = {0, 1};
vector<int> worktableIndexSellArray = {1, 2};
int robotId = 0;
double minDistance;
int minWorktableIndex, minWorktableSellIndex;

// 调用函数
findShortestBuyDistance(worktableIndexBuyArray, worktableIndexSellArray,
                        worktable_array, robot_array,
                        robotId, minDistance, minWorktableIndex,
                        minWorktableSellIndex);

// 打印结果
cout << "最短距离: " << minDistance << endl;
cout << "购买工作台: " << minWorktableIndex << endl;
cout << "出售工作台: " << minWorktableSellIndex << endl;
```

在这个例子中，函数会计算机器人从 (5, 5) 到工作台 0 或 1 的距离，然后分别计算工作台 0 或 1 到工作台 1 或 2 的距离，最终找到距离和最小的组合。

**2. `isItemEnough` 函数**

```c++
/**
 * 物品worktable_type够了返回true,不够返回false
 * @param worktable_array
 * @param worktable_type
 * @return
 */
bool isItemEnough(vector<worktable> &worktable_array, int worktable_type) {
    int sumSupply = 0;//统计供给和
    int sumDemand = 0;//统计需求和
    int i;
    for (i = 0; i < worktable_array.size(); i++) {
        if (worktable_array[i].worktable_type == worktable_type) {
            if (worktable_array[i].remainingProductionTime != -1 ||
                worktable_array[i].productCellStatus == 1) {//说明在生产或者有了产品
                sumSupply++;//统计供给和
            }
        }
        if (worktable_array[i].worktable_type == 7) {
            if (worktable_array[i].raw_material_cell[worktable_type] == '0') {//7号工作台缺该物品
                sumDemand++;//统计需求和
            }
        }
    }
    if (sumSupply >= sumDemand) {//如果供给大于等于需求,那就不需要供给,
        return true;
    } else {
        return false;
    }
}
```

**描述:**

这个函数用于判断某种物品 `worktable_type` 在当前环境中是否足够。 它通过比较该物品的供给量和需求量来实现。

**使用说明:**

1.  **输入:**
    *   `worktable_array`: 存储了所有工作台信息的数组。
    *   `worktable_type`:  指定要判断的物品类型。

2.  **流程:**
    *   初始化 `sumSupply` 和 `sumDemand` 为 0。
    *   遍历所有工作台。
    *   如果工作台的类型是 `worktable_type`，并且该工作台正在生产或者已经生产出了该物品（`remainingProductionTime != -1 || productCellStatus == 1`），则增加 `sumSupply`。
    *   如果工作台的类型是 7（7号工作台代表最终产品组装台），并且该工作台缺少 `worktable_type` 物品（`raw_material_cell[worktable_type] == '0'`），则增加 `sumDemand`。
    *   如果 `sumSupply` 大于等于 `sumDemand`，则返回 `true`，否则返回 `false`。

3.  **输出:**
    *   `true` 如果物品足够，`false` 如果物品不够。

**简单示例:**

```c++
vector<worktable> worktable_array = {
    {4, -1, 1}, // 工作台 0，类型 4，正在生产
    {5, -1, 1}, // 工作台 1，类型 5，正在生产
    {7, {'0', '1', '0'}, 0}  // 工作台 2，类型 7，缺少物品 0 和 2
};

int worktable_type = 0;

// 调用函数
bool isEnough = isItemEnough(worktable_array, worktable_type);

// 打印结果
cout << "物品是否足够: " << (isEnough ? "是" : "否") << endl; //输出否
```

在这个例子中，由于只有一个类型为 7 的工作台缺少物品 0，但是没有类型为 0 的工作台提供物品 0，所以函数返回 `false`。

**3. `findMaxPriority` 函数**

```c++
/**
 * 计算买物品的优先级,函数用于计算优先级
 * @param worktableIndexBuyArray 工作台id数组,存储在当前帧,能被机器人robotId所选择并且成品格有物品i的所有工作台的id.(该工作台的产品格有物品i,并且没有其他机器人标记该产品格)
 * @param worktableIndexSellArray 工作台id数组,存储在当前帧,能收购物品i的所有工作台的id(物品对应的该工作台的原材料格是空的,并且没有其他机器人标记该原材料格)
 * @param worktable_array 工作台数组,存储所有工作台信息
 * @param robot_array 机器人数组,存储所有机器人信息
 * @param robotId 机器人id
 * @param minDistance 存储最短距离和,和为机器人到工作台id=minWorktableIndex的距离平方+工作台id=minWorktableIndex的距离到工作台id=minWorktableSellIndex的距离平方
 * @param minWorktableIndex 买该物品的工作台id
 * @param minWorktableSellIndex 卖该物品的工作台id
 * @param productPriorityArray 存储每种物品的优先级
 * @param itemId 物品号-1
 * @param worktable_array7 存储所有7号工作台
 * 通过& 返回距离和minDistance,买物品的工作台id=minWorktableIndex,卖物品的工作台id=minWorktableSellIndex,该物品的优先级priority
 */
void
findMaxPriority(vector<int> &worktableIndexBuyArray, vector<int> &worktableIndexSellArray,
                vector<worktable> &worktable_array, vector<robot> &robot_array,
                int &robotId, double &minDistance, int &minWorktableIndex,
                int &minWorktableSellIndex, vector<productPriority> &productPriorityArray, int &itemId,
                vector<worktable> &worktable_array7, int &frameID) {//寻找最近的买卖距离和
    double distanceBetween, distanceBetweenPoint, priority;
    double maxPriority = 0;
    int i, j;
    if (worktable_array7.size() > 2) {//专门处理心形
        for (i = 0; i < worktableIndexBuyArray.size(); i++) {
            distanceBetweenPoint = distanceSquaredBetweenPoint(robot_array[robotId],
                                                               worktable_array[worktableIndexBuyArray[i]]);
            priority = 1000.0 / distanceBetweenPoint;
            if (maxPriority < priority) {//如果是最短路径,记录距离和走过的两个结点
                maxPriority = priority;
                minWorktableIndex = worktableIndexBuyArray[i];
            }
        }
        if (worktable_array[minWorktableIndex].worktable_type >= 4 &&
            worktable_array[minWorktableIndex].worktable_type <= 6) {//如果卖给7号
            maxPriority = 0;
            for (j = 0; j < worktableIndexSellArray.size(); j++) {
                if (worktable_array[worktableIndexSellArray[j]].worktable_type ==
                    7) {//不区分是否在生产,选择缺原材料少的工作台
                    if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState == 0) {//缺三种
                        distanceBetween = 10000;
                    } else if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                               48 ||
                               worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                               80 ||
                               worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                               96) {//缺一种,取了中间值
                        distanceBetween = 30000;
                    } else {//缺两种,取了中间值
                        distanceBetween = 20000;
                    }
                }
                if (maxPriority < distanceBetween) {//记录下缺原材料最少的工作台
                    maxPriority = distanceBetween;
                    minWorktableSellIndex = worktableIndexSellArray[j];
                }
            }
        } else {//如果卖给456或89,选择最近的
            for (j = 0; j < worktableIndexSellArray.size(); j++) {
                distanceBetween = distanceSquaredBetweenWorktable(worktable_array[minWorktableIndex],
                                                                  worktable_array[worktableIndexSellArray[j]]);
                if (minDistance > distanceBetween) {
                    minDistance = distanceBetween;
                    minWorktableSellIndex = worktableIndexSellArray[j];
                }
            }
        }
        productPriorityArray[itemId].priority = maxPriority;
        productPriorityArray[itemId].minWorktableIndex = minWorktableIndex;
        productPriorityArray[itemId].minWorktableSellIndex = minWorktableSellIndex;

        return;
    }
    for (i = 0; i < worktableIndexBuyArray.size(); i++) {
        distanceBetweenPoint = distanceSquaredBetweenPoint(robot_array[robotId],
                                                           worktable_array[worktableIndexBuyArray[i]]);
        for (j = 0; j < worktableIndexSellArray.size(); j++) {

            distanceBetween = distanceSquaredBetweenWorktable(worktable_array[worktableIndexBuyArray[i]],
                                                              worktable_array[worktableIndexSellArray[j]]);
//            priority = productPriorityArray[itemId].profit /
//                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));//买卖物品产生的优先级
            if (worktable_array[worktableIndexSellArray[j]].worktable_type >= 4 &&
                worktable_array[worktableIndexSellArray[j]].worktable_type <= 6) {//如果卖给456
                if (worktable_array[worktableIndexSellArray[j]].worktable_type == 4 &&
                    worktable_array7.size() == 1&&frameID<7321) {//箭头形
                    priority = 99999999 /
                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                } else {
                    if (worktable_array[worktableIndexSellArray[j]].remainingProductionTime ==
                        -1) {
                        int index;
                        for (index = 0; index < worktable_array7.size(); index++) {//遍历数组中所有的7号工作台
                            if (worktable_array[worktable_array7[index].worktableArrayIndex].remainingProductionTime ==
                                -1) {//处于未生产状态中,说明需要材料
                                //如果所有7号生产需要该物品x个,并且没有地图没有x个该物品,那就再加成
                                if (!isItemEnough(worktable_array,
                                                  worktable_array[worktableIndexSellArray[j]].worktable_type) &&
                                    worktable_array[worktable_array7[index].worktableArrayIndex].raw_material_cell[worktable_array[worktableIndexSellArray[j]].worktable_type] ==
                                    '0') {
                                    //如果这个7号生产需要该物品,那就再加成
                                    if (worktable_array[worktableIndexSellArray[j]].worktable_type == 4 &&
                                        worktable_array[worktable_array7[index].worktableArrayIndex].rawMaterialCellState ==
                                        96) {//7号二缺一,缺4,减去物品5的和14200与物品6的和14900
                                        if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                            0) {//说明缺两种
                                            priority = (71400 - 14200 - 14900) / 4.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        } else {
                                            priority = ((71400 - 14200 - 14900) / 4.0 * 3.0 - 3200) / 2.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        }
                                    } else if (worktable_array[worktableIndexSellArray[j]].worktable_type == 5 &&
                                               worktable_array[worktable_array7[index].worktableArrayIndex].rawMaterialCellState ==
                                               80) {//7号二缺一,缺5,减去物品4的和13300与物品6的和14900
                                        if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                            0) {//说明缺两种
                                            priority = (71400 - 13300 - 14900) / 4.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        } else {
                                            priority = ((71400 - 13300 - 14900) / 4.0 * 3.0 - 3200) / 2.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        }
                                    } else if (worktable_array[worktableIndexSellArray[j]].worktable_type == 6 &&
                                               worktable_array[worktable_array7[index].worktableArrayIndex].rawMaterialCellState ==
                                               48) {//7号二缺一,缺6,减去物品4的和13300与物品5的和14200
                                        if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                            0) {//说明缺两种
                                            priority = (71400 - 13300 - 14900) / 4.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        } else {
                                            priority = ((71400 - 13300 - 14900) / 4.0 * 3.0 - 3200) / 2.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        }
                                    } else if (
                                            worktable_array[worktable_array7[index].worktableArrayIndex].rawMaterialCellState ==
                                            0) {//7号缺三种
                                        if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                            0) {//说明缺两种
                                            priority = 71400.0 / (6 + 3 + 1) /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        } else {
                                            priority = (71400.0 / (6 + 3 + 1) * 3.0 - 3200) / 2.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        }
                                    } else {//7号缺两种,因为缺两种的计算太复杂了,直接用中位数来代替
                                        if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                            0) {//说明缺两种
                                            priority = (71400 - 14200) / 7.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        } else {
                                            priority = ((71400 - 14200) / 7.0 * 3.0 - 3200) / 2.0 /
                                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        if (index ==
                            worktable_array7.size()) {//如果没有获得7的加成,那就只能获得物品4物品5物品6的加成,判断的方法是是否有break,如果break,那index不等于worktable_array7.size()
                            if (worktable_array[worktableIndexSellArray[j]].worktable_type == 4) {
                                if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState == 0) {//说明缺两种
                                    priority = (3000 + 3200 + 7100) / 3.0 /
                                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                } else {//取了平均值
                                    priority = (3100 + 7100) /2.0/
                                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                }
                            } else if (worktable_array[worktableIndexSellArray[j]].worktable_type == 5) {
                                if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState == 0) {//说明缺两种
                                    priority = (3000 + 3400 + 7800) / 3.0 /
                                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                } else {//取了平均值
                                    priority = (3200 + 7800) /2.0/
                                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                }
                            } else if (worktable_array[worktableIndexSellArray[j]].worktable_type == 6) {
                                if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState == 0) {//说明缺两种
                                    priority = (3200 + 3400 + 7800) / 3.0 /
                                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                } else {//取了平均值
                                    priority = (3300 + 7800) /2.0/
                                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                                }
                            }
                        }
                    } else {//如果在生产,那将没有优先级加成
                        priority = productPriorityArray[itemId].profit /
                                   (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));//买卖物品产生的优先级
                    }
                }
            } else if (worktable_array[worktableIndexSellArray[j]].worktable_type == 7) {//如果卖给7号
                if (worktable_array[worktableIndexBuyArray[i]].worktable_type == 4 &&
                    worktable_array7.size() == 1&&frameID<7321) {//箭头形
                    priority = 99999999 /
                               (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                } else {
                    //如果七号工作台没有生产,那就可以获得加成
                    if (worktable_array[worktableIndexSellArray[j]].remainingProductionTime <=
                        remainingProductionTime) {//没有在生产,或者快生产完,就给他
                        if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState == 0) {//缺三种
                            priority = (71400 / (6 + 3 + 1) * 3.0 -
                                        (3000 + 3200 + 3400 - productPriorityArray[itemId].profit)) /
                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                        } else if (worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                   48 ||
                                   worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                   80 ||
                                   worktable_array[worktableIndexSellArray[j]].rawMaterialCellState ==
                                   96) {//缺一种,取了中间值
                            priority = hyperParameterPriorityMap1 * hyperParameterPriorityMap1 *
                                       ((71400 - 14900 - 13300) / 4.0 * 3.0 - 3400 - 3000) /
                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                        } else {//缺两种,取了中间值
                            priority = hyperParameterPriorityMap1 * ((71400 - 14900) / 7.0 * 3.0 - 3000 - 3200) /
                                       (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));
                        }
                    } else {//没有七号的加成
                        priority = productPriorityArray[itemId].profit /
                                   (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));//买卖物品产生的优先级
                    }
                }

            } else {//如果卖给8或9,没有加成
                priority = productPriorityArray[itemId].profit /
                           (sqrt(distanceBetweenPoint) + sqrt(distanceBetween));//买卖物品产生的优先级
            }



            if (maxPriority < priority) {//如果是最短路径,记录距离和走过的两个结点

                maxPriority = priority;
                minWorktableSellIndex = worktableIndexSellArray[j];
                minWorktableIndex = worktableIndexBuyArray[i];
            }
        }
    }//保存遍历的结果
    productPriorityArray[itemId].priority = maxPriority;
    productPriorityArray[itemId].minWorktableIndex = minWorktableIndex;
    productPriorityArray[itemId].minWorktableSellIndex = minWorktableSellIndex;
}
```

**描述:**

`findMaxPriority` 函数用于为指定的机器人 `robotId` 计算购买某个物品 `itemId` 的最佳优先级。它考虑了多种因素，包括距离、物品利润、以及是否能够帮助 7 号工作台生产最终产品。这个函数目的是在一个复杂的场景下，通过优先级计算，让机器人做出最有利于整体生产的决策。

**使用说明:**

1.  **输入:**
    *   `worktableIndexBuyArray`: 可以购买物品 `itemId` 的工作台 ID 列表。
    *   `worktableIndexSellArray`: 可以出售物品 `itemId` 的工作台 ID 列表。
    *   `worktable_array`: 所有工作台信息的数组。
    *   `robot_array`: 所有机器人信息的数组。
    *   `robotId`:  要计算优先级的机器人 ID。
    *   `minDistance`: (输入输出参数) 存储最小距离和，但在这个函数中，它更多的是在特定的分支中使用，用于辅助决策。
    *   `minWorktableIndex`: (输出参数) 存储购买工作台的 ID。
    *   `minWorktableSellIndex`: (输出参数) 存储出售工作台的 ID。
    *   `productPriorityArray`:  存储每种物品的基础优先级信息，比如利润。
    *   `itemId`:  物品的 ID。
    *   `worktable_array7`: 存储 7 号工作台的信息。
    *   `frameID`: 当前帧的 ID，可能用于根据时间调整策略。

2.  **流程:**

    *   **心形处理 (worktable\_array7.size() > 2):**  如果存在多个 7 号工作台（心形布局），则采用一种特殊的策略，优先考虑距离机器人最近的购买工作台，并且优先将物品卖给缺少原材料的 7 号工作台。
    *   **常规优先级计算:**  遍历所有可能的购买和出售工作台组合，计算每个组合的优先级。
        *   如果将物品卖给 4, 5, 6 号工作台，则检查这些工作台是否处于空闲状态（`remainingProductionTime == -1`），如果是，则根据 7 号工作台的需求情况进行优先级加成。如果 7 号工作台缺少该物品，则提高优先级。
        *   如果将物品卖给 7 号工作台，则根据 7 号工作台的生产状态（是否在生产，缺少哪些原材料）进行优先级加成。
        *   如果将物品卖给 8 或 9 号工作台，则没有额外的优先级加成，优先级仅与物品的利润和距离有关。
    *   更新 `productPriorityArray[itemId]`，存储计算出的最高优先级以及对应的购买和出售工作台 ID。

3.  **输出:**

    *   通过引用传递，`productPriorityArray[itemId]`，`minWorktableIndex` 和 `minWorktableSellIndex` 将被更新为找到的最佳值。

**简单示例:**

假设你有以下数据:

```c++
vector<worktable> worktable_array = {
    {4, -1, 1, 10.0, 10.0, {'0', '1', '1'}, 0},   // 工作台 0，类型 4，正在生产
    {7, -1, 0, 20.0, 20.0, {'0', '0', '0'}, 0},    // 工作台 1，类型 7，缺少所有原材料
    {5, -1, 1, 30.0, 30.0, {'1', '1', '1'}, 0}    // 工作台 2，类型 5，正在生产
};

vector<robot> robot_array = {
    {5.0, 5.0}   // 机器人 0
};

vector<int> worktableIndexBuyArray = {0, 2};     // 可以购买物品 4 和 5
vector<int> worktableIndexSellArray = {1};      // 可以卖给 7 号工作台
int robotId = 0;
double minDistance = 99999.0;
int minWorktableIndex, minWorktableSellIndex;
vector<productPriority> productPriorityArray(7);
productPriorityArray[3].profit = 7100.0;       // 物品 4 的利润
int itemId = 3;                             // 要购买物品 4
vector<worktable> worktable_array7 = {worktable_array[1]}; //7号工作台信息
int frameID = 1;

// 调用函数
findMaxPriority(worktableIndexBuyArray, worktableIndexSellArray,
                worktable_array, robot_array,
                robotId, minDistance, minWorktableIndex,
                minWorktableSellIndex, productPriorityArray, itemId,
                worktable_array7, frameID);

// 打印结果
cout << "最高优先级: " << productPriorityArray[itemId].priority << endl;
cout << "购买工作台: " << productPriorityArray[itemId].minWorktableIndex << endl;
cout << "出售工作台: " << productPriorityArray[itemId].minWorktableSellIndex << endl;
```

在这个例子中，函数会计算机器人购买物品 4 并卖给 7 号工作台的优先级。 由于 7 号工作台缺少所有原材料，并且物品 4 可以帮助 7 号工作台生产，因此函数会提高购买物品 4 的优先级。

**4. `isEnoughTimeBuy` 函数**

```c++
/**
 * 判断剩余时间是否足够买卖物品,判断思路为最少花费帧数*超参数系数hyperParameterTimeBuy>剩余帧数,说明时间不够(其实小于也不能说明时间够)
 * @param robot1 机器人
 * @param worktableBuy 买物品的工作台
 * @param worktableSell 卖物品的工作台
 * @param frameId 当前帧数
 * @return 时间够返回true,时间不够返回false
 */
bool isEnoughTimeBuy(robot &robot1, worktable &worktableBuy, worktable &worktableSell, int frameId) {//判读时间是否够买东西
    if (9000 - frameId < hyperParameterTimeBuy * 50.0 * (sqrt(distanceSquaredBetweenPoint(robot1, worktableBuy)) +
                                                         sqrt(distanceSquaredBetweenWorktable(worktableBuy,
                                                                                              worktableSell))) /
                         6.0) {//如果时间肯定不够,就不买
        return false;
    } else
        return true;
}
```

**描述:**

这个函数用于判断在给定的剩余时间内，机器人是否足够时间完成一次买卖物品的交易。 它考虑了机器人到购买工作台的距离，购买工作台到出售工作台的距离，以及一个超参数 `hyperParameterTimeBuy` 用于调整时间估计。

**使用说明:**

1.  **输入:**
    *   `robot1`: 机器人信息。
    *   `worktableBuy`: 购买工作台信息。
    *   `worktableSell`: 出售工作台信息。
    *   `frameId`:  当前帧的 ID。

2.  **流程:**
    *   计算剩余时间 `9000 - frameId` (假设总帧数为 9000)。
    *   计算机器人到购买工作台的距离，以及购买工作台到出售工作台的距离。
    *   使用超参数 `hyperParameterTimeBuy` 调整时间估计。
    *   如果估计的时间大于剩余时间，则返回 `false`，否则返回 `true`。

3.  **输出:**
    *   `true` 如果有足够的时间，`false` 如果时间不够。

**简单示例:**

```c++
robot robot1 = {5.0, 5.0};
worktable worktableBuy = {4, -1, 1, 10.0, 10.0};
worktable worktableSell = {7, -1, 0, 20.0, 20.0};
int frameId = 1000;
double hyperParameterTimeBuy = 1.0;

// 调用函数
bool isEnough = isEnoughTimeBuy(robot1, worktableBuy, worktableSell, frameId);

// 打印结果
cout << "时间是否足够: " << (is