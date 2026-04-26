# Group Project Proposal 中文参考译文

组号：Group [TBD]  
组员：[Member 1], [Member 2], [Member 3]  
课程：BU.520.750.51.SP26  
日期：2026 年 4 月 26 日

## Title

基于表格型强化学习的零售 SKU 自适应库存与定价决策

## Problem & "Why RL?"

零售经理需要反复决定补多少货、设什么价格。一次性预测模型可以预测需求，但不能直接优化会影响未来状态的行动。今天打折可能增加销量，但也会提高明天缺货的风险；今天大量补货可能保障服务水平，但如果需求下降就会产生持有成本。因此，这是一个不确定环境下的序贯决策问题，agent 需要学习长期策略，在利润率、商品可得性和库存成本之间取得平衡。

## MDP Formulation

状态空间：我们每天观察一个有限元组：`(inventory_bin, demand_signal_bin, price_tier, calendar_type, pipeline_bin)`。库存被离散为缺货、低、中、高、过量；需求信号基于近期销量分位数离散为低、正常、高；价格离散为折扣价、常规价、溢价；日历分为工作日、周末、事件日和 SNAP 相关日；在途补货分为无、小、大。

动作空间：每个动作都是离散的：`(price_choice, order_quantity)`。价格选择为折扣价、常规价、溢价；订货量为不订、小批量、中批量、大批量。

奖励函数：每日奖励等于运营利润，即销售收入减去采购成本、固定订货成本、持有成本、缺货惩罚和过量库存惩罚。这样既鼓励有利润的销售，又避免长期缺货或不必要的过量库存。

方法：核心方法将使用 tabular Q-learning 和 SARSA。如果时间允许，会在估计出的转移模型上增加 Value Iteration 或 Policy Iteration 对比。Deep RL 和连续函数近似不会作为主要方法。

## Environment and Data Strategy

我们将构建一个由公开零售数据校准的有限状态模拟器。主数据集为 Kaggle 的 M5 Forecasting - Accuracy Walmart 数据集，其中包含商品-门店日销量、销售价格和日历事件。由于数据中没有真实库存，我们会在模拟器中显式设定库存动态、提前期、单位成本、持有成本和缺货惩罚。项目会从一个门店-SKU 或小 SKU 集合开始，将所有变量离散化为有限分箱，并从经验上下文分布中抽样需求。较小的 Kaggle 零售数据集会作为原型开发的备选数据源。

## Baselines and Evaluation

我们会将 RL agent 与固定价格加 `(s, S)` 补货策略、常规价格加平均需求目标库存策略、简单 markdown 启发式策略以及随机合法动作策略进行比较。评估将使用留出的模拟 episode 和多个随机种子。主指标是平均累计利润；辅助指标包括 fill rate、缺货天数、平均期末库存、库存周转和价格变动次数。学习曲线和策略热力图将展示 tabular agent 是否学到了可解释的行为。

致谢：本 proposal 使用了课程材料、公开数据集文档、关于 Q-learning 定价/库存的研究文章，以及 ChatGPT/Codex 的 GenAI 辅助来进行草拟和组织。
