# 项目规划：基于表格型强化学习的自适应库存与定价

## 1. 执行摘要

本项目研究一个贴近零售运营的序贯决策问题：门店每天需要同时决定某个商品或小品类的补货数量和价格水平。这两个决策相互影响。较高价格可能提高单位利润但降低需求；较低价格可能提升销量但增加缺货风险；激进补货可以提高可得性但也会增加持有成本。项目目标是学习一个自适应策略，在长期利润和服务水平之间取得更优平衡。

为了满足课程范围要求，本项目会使用有限、离散的状态空间和动作空间，并采用表格型强化学习方法。核心算法为 Q-learning 和 SARSA。如果时间允许，还会从模拟器估计转移模型，并用 Value Iteration 或 Policy Iteration 做对比。

## 2. 研究动机

库存管理和动态定价都是典型的序贯决策问题。普通预测模型可以预测需求，但不能直接回答“今天应该采取什么行动”，尤其当今天的行动会改变明天的库存状态和未来收益时。今天打折可能减少库存并增加之后的缺货概率；今天大量订货可能降低缺货损失，但如果需求下降则会带来库存持有成本。这种反馈结构使强化学习非常适合该问题。

相关研究也支持这种建模方式。近年的零售定价研究使用 Q-learning 进行自适应定价；库存管理研究通常把补货问题建模为 MDP，其中包含库存水平、提前期、订货成本、持有成本和缺货惩罚。M5 Forecasting 数据集提供了较可靠的零售基础，因为它包含 Walmart 的商品-门店日销量、价格和日历特征。

## 3. 推荐研究范围

为了保持项目可控，并且清楚体现 tabular RL，第一版实现建议聚焦于 M5 数据中的一个门店和 1 到 5 个 SKU。优先选择销量较稳定的食品或家居商品。只有当单 SKU 环境跑通之后，再考虑扩展到小品类或多 SKU。

推荐第一版设定：

- 一个决策周期为一天。
- 核心实验使用一个门店和一个 SKU。
- 库存容量限制为较小整数，例如 40 或 60 件。
- 补货提前期设为 1 或 2 天。
- 三个价格档位：折扣价、常规价、溢价。
- 四个订货水平：不订、小批量、中批量、大批量。
- 状态特征全部离散化为有限元组。

这种范围可以让 Q-table 足够小，方便调试、可视化和课堂展示。

## 4. 数据策略

### 主数据集：M5 Forecasting - Accuracy

M5 是最推荐的主数据集，因为它公开、使用广泛，并且基于 Walmart 零售销售数据。它包含：

- 商品-门店级别的日销量。
- 商品层级信息，例如品类和部门。
- 门店和州信息。
- 每周销售价格。
- 日历特征和特殊事件。

该数据集不包含真实的在手库存和采购成本。因此，我们会用它校准需求和价格上下文，再构建一个自定义模拟器，在模拟器中显式设定库存动态和成本参数。这符合课程允许使用模拟器或真实数据的要求，因为我们的环境会以真实零售需求模式为基础。

### 备选数据集 1：Store Item Demand Forecasting Dataset

这个 Kaggle 数据集规模较小且为合成数据，包含每日门店-商品销售、价格、促销、星期和月份。由于价格和促销变量已经直接给出，它适合快速原型开发。

### 备选数据集 2：Retail Sales Promotions and Demand Forecasting

这个 Kaggle 合成数据集包含销售、折扣、价格、促销、库存水平和时间因素。如果 M5 下载或预处理成本过高，它可以作为快速演示数据源。

## 5. MDP 建模

### 状态空间

第 `t` 天的状态定义为有限元组：

`s_t = (inventory_bin, demand_signal_bin, price_tier, calendar_type, pipeline_bin)`

候选离散分箱如下：

- `inventory_bin`：缺货、低库存、中库存、高库存、过量库存。
- `demand_signal_bin`：低、正常、高，基于近 7 天或 14 天需求分位数。
- `price_tier`：折扣价、常规价、溢价。
- `calendar_type`：工作日、周末、节假日/事件日、SNAP 或事件相关日。
- `pipeline_bin`：无在途订单、小订单即将到达、大订单即将到达。

这样大约产生 `5 x 3 x 3 x 4 x 3 = 540` 个状态。配合 12 到 15 个动作，Q-table 仍然足够小，适合表格型学习和解释。

### 动作空间

每个动作同时包含价格决策和订货决策：

`a_t = (price_choice, order_quantity)`

候选离散动作如下：

- `price_choice`：折扣价、常规价、溢价。
- `order_quantity`：0、小包装、中包装、大包装。

例如，动作 `(discount, medium)` 表示今天设置折扣价，并下一个中等批量的补货订单。

### 奖励函数

奖励定义为每日运营利润，并加入服务水平惩罚：

`reward = revenue - procurement_cost - fixed_order_cost - holding_cost - stockout_penalty - excess_inventory_penalty`

其中：

- `revenue = selling_price x fulfilled_demand`
- `procurement_cost = unit_cost x order_quantity`
- `holding_cost = h x ending_inventory`
- `stockout_penalty = p x lost_sales`

这种奖励设计鼓励 agent 在利润率、可得性和库存效率之间平衡。它避免只奖励收入导致过度订货，也避免只奖励低库存导致持续缺货。

## 6. 环境设计

环境将是一个由零售数据校准的自定义表格型模拟器：

1. 从 M5 中选择一个有足够非零需求和价格变化的门店-SKU 序列。
2. 将日销量、价格、星期、事件和近期需求历史转换为离散上下文。
3. 按上下文估计经验需求分布。如果某个分箱样本过少，则回退到更宽的品类级或门店级经验分布。
4. 每个模拟日执行以下步骤：
   - agent 观察离散状态。
   - agent 选择价格档位和订货水平。
   - 根据校准的经验需求分布，并加入简单价格档位调整，抽样当天需求。
   - 销量不超过可用库存。
   - 计算持有、缺货、采购和订货成本。
   - 在提前期之后收到在途补货。
5. 每个 episode 固定为 90 或 180 天。

该模拟器足够简单，便于课堂解释，同时又能体现定价和补货之间的真实权衡。

## 7. 强化学习方法

核心方法：

- Tabular Q-learning：off-policy 学习，使用 epsilon-greedy 探索。
- Tabular SARSA：on-policy 对照，用于观察在探索存在时是否学到更保守的策略。

可选方法：

- 基于模拟器估计转移和奖励模型，再运行 Value Iteration。
- 在同一有限 MDP 上运行 Policy Iteration。
- 做一个只优化价格的 Multi-Armed Bandit 消融实验，用来呼应 Week 2，并说明库存状态的重要性。

训练设置：

- 折扣因子 `gamma`：0.90 到 0.99。
- 学习率 `alpha`：固定或衰减，并做网格比较。
- 探索率 `epsilon`：从较高值开始并逐步衰减。
- 随机种子：至少使用 10 个模拟种子提升评估稳定性。
- 超参数选择：基于验证 episode，而不是测试集表现。

本项目不会把 deep RL 或连续函数近似作为主要方法。

## 8. 基线策略

训练后的 RL 策略应与清晰的非 RL 基线比较：

- 固定价格加 `(s, S)` 补货策略。
- 固定常规价格加按平均需求补到目标库存。
- 启发式降价策略：高库存时打折，低库存时溢价。
- 随机合法动作策略。
- 可选：根据 M5 历史价格档位回放的策略。

其中 `(s, S)` 基线尤其重要，因为它是标准库存策略，且容易解释。

## 9. 评估指标

主指标：

- 在留出模拟期间的平均 episode 累计利润。

辅助指标：

- Fill rate：满足需求量除以总需求量。
- 缺货天数。
- 平均期末库存。
- 库存周转。
- 价格稳定性或价格变动次数。
- 对持有成本、缺货惩罚和提前期的敏感性。

最终报告应包含学习曲线、策略热力图以及 RL 与基线对比表。

## 10. 计划交付物

最低可行最终项目：

- 干净的数据预处理 notebook 或脚本。
- 自定义有限 MDP 模拟器。
- Q-learning 和 SARSA 实现。
- 基线策略。
- 带图表和表格的评估 notebook。
- 最终展示中的策略解释。

更强版本：

- 加入 Value Iteration 或 Policy Iteration 对比。
- 针对提前期和成本设置做敏感性分析。
- 小规模多 SKU 或品类级实验。
- 做一个交互式 demo，展示学到的策略如何响应库存和需求状态。

## 11. 时间线

第 6 周：

- 完成 proposal。
- 确认组内分工。
- 下载样本数据并选择候选 SKU/门店。

第 7 周：

- 构建预处理流程。
- 定义离散状态/动作空间。
- 实现模拟器和基线策略。

第 8 周：

- 实现 Q-learning 和 SARSA。
- 跑第一轮实验。
- 调试奖励尺度和转移逻辑。

第 9 周：

- 调参。
- 添加评估指标和图表。
- 与基线比较。

第 10 周：

- 准备最终报告、代码清理和展示。
- 如时间允许，加入敏感性分析。

## 12. 风险与应对

风险：M5 不包含真实库存。

应对：用 M5 校准需求，在模拟器中显式设定库存动态。清楚说明单位成本、持有成本、缺货惩罚和提前期假设。

风险：历史价格不是随机实验，价格弹性存在因果识别不确定性。

应对：把价格响应作为模拟器场景，而不是因果结论。设置低、中、高三种价格弹性情景做敏感性分析。

风险：状态空间过大。

应对：从单 SKU、粗分箱和五个状态组件开始。只有当 tabular agent 跑通后再扩展。

风险：奖励函数可能诱导非预期行为。

应对：除了利润，还跟踪 fill rate、缺货天数和期末库存。调整惩罚参数并报告敏感性。

## 13. 建议组内分工

- 数据负责人：下载和预处理 M5 或备选数据集，选择 SKU/门店子集。
- 环境负责人：实现模拟器、状态离散化、转移和奖励函数。
- RL 负责人：实现 Q-learning、SARSA 和可选 Value Iteration。
- 评估与报告负责人：实现基线、图表、最终 proposal/report 写作和展示。

如果小组人数较少，可以合并数据和环境角色，并合并 RL 与评估角色。

## 14. 资料链接

- M5 Forecasting - Accuracy, Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- M5Dataset documentation, sktime: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.datasets.forecasting.m5_competition.M5Dataset.html
- Makridakis et al., M5 competition background: https://www.sciencedirect.com/science/article/pii/S0169207021001187
- Dynamic Retail Pricing via Q-Learning: https://arxiv.org/abs/2411.18261
- Hybrid algorithm based on reinforcement learning for smart inventory management: https://link.springer.com/article/10.1007/s10845-022-01982-5
- Inventory management MDP notes: https://adityam.github.io/stochastic-control/mdps/inventory-management-revisited.html
- Store Item Demand Forecasting Dataset: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Retail Sales Promotions and Demand Forecasting: https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting

## 15. 学术诚信说明

最终提交应致谢使用过的公开数据集、参考文献、课程材料，以及用于草拟、组织或改进 proposal 和实施计划的 GenAI 工具辅助。
