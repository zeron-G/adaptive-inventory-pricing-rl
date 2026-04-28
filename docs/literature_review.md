# Literature Review and Citation Notes

This project is positioned as a tabular reinforcement-learning implementation for a finite retail inventory-pricing simulator calibrated from public Walmart sales data.

## Reinforcement Learning Foundations

Sutton and Barto (2018) provide the standard treatment of Markov Decision Processes, temporal-difference learning, SARSA, and Q-learning. The project uses their finite-MDP framing: a state summarizes the decision-relevant information, actions change the next state distribution, and the objective is expected discounted cumulative reward.

Watkins and Dayan (1992) introduced and analyzed Q-learning for controlled Markovian domains. Q-learning is appropriate here because the simulator exposes a finite state-action table. SARSA is included as an on-policy comparison because it evaluates the consequences of exploratory behavior rather than only the greedy target action (Rummery & Niranjan, 1994; Sutton & Barto, 2018).

## Retail Data and M5

The M5 Forecasting Accuracy data is a public Walmart sales dataset from the M5 competition. The dataset includes daily unit sales, sell prices, and calendar features for thousands of product-store series (Kaggle, 2024; Makridakis et al., 2022; Zenodo, 2024). This makes it a strong basis for demand calibration. However, it does not provide true on-hand inventory, lost demand, or replenishment decisions. The project therefore does not claim to learn Walmart's historical inventory policy. Instead, M5 is used to calibrate a simulator.

## Inventory and Pricing as Sequential Decision Problems

Inventory control is naturally represented as a sequential decision problem because today's order affects future availability, holding cost, and stockout risk. Recent inventory-control literature describes RL as suitable for inventory problems with ordering, holding, and shortage costs (Gijsbrechts et al., 2022; NEASQC, 2024). Mahajan (n.d.) presents inventory management directly through MDP notation, which closely matches this project's formulation.

Dynamic pricing is also sequential: today's price affects immediate demand and future inventory. Apte et al. (2024) demonstrate a Q-learning framework for simulated retail pricing, supporting our choice of a tabular Q-learning baseline for price adaptation. Our project extends the idea by coupling discrete pricing with replenishment.

## APA References

Apte, M., Kale, K., Datar, P., & Deshmukh, P. (2024). *Dynamic retail pricing via Q-learning: A reinforcement learning framework for enhanced revenue management*. arXiv. https://arxiv.org/abs/2411.18261

Gijsbrechts, J., Boute, R. N., Van Mieghem, J. A., & Zhang, D. (2022). Deep reinforcement learning for inventory control: A roadmap. *European Journal of Operational Research, 298*(2), 401-412. https://doi.org/10.1016/j.ejor.2021.07.016

Kaggle. (2024). *M5 Forecasting - Accuracy*. https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Mahajan, A. (n.d.). *Inventory management revisited*. https://adityam.github.io/stochastic-control/mdps/inventory-management-revisited.html

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). The M5 competition: Background, organization, and implementation. *International Journal of Forecasting, 38*(4), 1325-1336. https://doi.org/10.1016/j.ijforecast.2021.07.007

NEASQC. (2024). *Reinforcement learning for inventory management*. https://www.neasqc.eu/use-case/reinforcement-learning-for-inventory-management/

Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems* (Technical Report CUED/F-INFENG/TR 166). Cambridge University Engineering Department.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*, 279-292. https://doi.org/10.1007/BF00992698

Zenodo. (2024). *M5 Forecasting Accuracy dataset* (Version v1) [Data set]. https://doi.org/10.5281/zenodo.12636070
