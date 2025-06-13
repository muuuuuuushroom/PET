## 6.13
- Exploring the original PET architecture on crowd counting
    - Dataset: SHA
    - Backbone: VGG
    - Parameters: None-changed hp, bs8 lr1e-4 ep1500

- Few exps are conducted
    - Baseline on VGG
    - New Training branch: Loss of Feature(4x) aligning with Probability map
    - Loss for points transfering to Loss Probs from Smooth L1
    - A mixture of the two above

- Hyper-para
    - Sigma for Probability map generation, 0.4 by default
    - F4X: prob_loss_coef, 1.0 initially
    - Probloss: raw loss scale factor, 0.05 initially
