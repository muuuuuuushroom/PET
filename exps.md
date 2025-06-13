## 6.13
- Exploring the original PET architecture on crowd counting
    - Dataset: SHA
    - Backbone: VGG16_bn
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
    - Probloss: gamma for facal loss, 2.0 initially

- Additional THOUGHTS:
    - thrs for split? fixed 0.5? dynamic?
    - using the output of 1st layer as the guidance for split
    - the other representation of the Eu-dis now in probloss
        ![Losses](./cache/image.png){ width=500 height=500 }
        - y = 1-x
        - y = -log(x)
        - y = 1-x^2
        - y = (1-x)^2
        - y = -(1-x)^k*log(x)
