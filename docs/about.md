# 项目与作者

cuButterfly 是一个面向 GPU 的 CUDA 研究库和可复现实验工件，研究对象是
FFT、NTT、FWHT、结构化 2x2 以及 subset/superset zeta 中的规则蝶形计算。
项目的主要贡献边界是依赖驱动的分层混合数据流、可替换计算核心、硬件约束下的
lowering 与测量选择。某个局部 arithmetic core 的实现技巧不自动归因于整个方法。

## 作者与归属

| 项目 | 信息 |
|---|---|
| 作者 / 维护者 | **Teng Wang** |
| 实验室 | High Efficient Intelligent Computing Lab |
| 机构 | Suzhou Institute for Advanced Research, University of Science and Technology of China |
| 所在地 | Suzhou, China |
| 邮箱 | [wangt635@ustc.edu.cn](mailto:wangt635@ustc.edu.cn) |
| GitHub | [TruNcat3](https://github.com/TruNcat3) |

项目由作者独立维护。机构名称用于说明研究归属，不表示中国科学技术大学或其苏州高等研究院
对本项目作官方发布、认证、赞助或背书。第三方代码、论文和软件的作者信息仍按各自许可证和
上游声明保留，详见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## 方法来源与贡献边界

cuButterfly 的方法分析来自共同蝶形依赖、stage/data 的时空展开、局部驻留和边界布局交接。
本仓库也可以吸收 cuFFTDx、VkFFT、TurboFFT、Dao FHT 或 GPU-NTT 等成熟 arithmetic core，
但必须把核心替换、组织替换和硬件 lowering 分开测量。跨 GPU 的 profile 和 selector 是目标卡
上的迁移证据，不是把一张卡的表格当成所有 GPU 的通用最优解。

## 引用

使用软件、实验数据或实现时，请引用 `CITATION.cff` 中的机器可读元数据。对应 BibTeX 为：

```bibtex
@software{wang_cubutterfly_2026,
  author  = {Teng Wang},
  title   = {cuButterfly: Hardware-Mapped Space-Time Parallelism for Butterfly Computations on GPUs},
  year    = {2026},
  version = {0.9.0},
  url     = {https://github.com/TruNcat3/cuButterfly},
  organization = {High Efficient Intelligent Computing Lab,
                  Suzhou Institute for Advanced Research,
                  University of Science and Technology of China}
}
```

论文工作稿仍按投稿准备保留匿名署名，见 [`paper/main.tex`](../paper/main.tex)。这只是论文的
匿名化边界，不改变软件和研究工件的作者元数据。

## 联系与复现

缺陷、复现问题和协议不一致优先通过
[GitHub Issues](https://github.com/TruNcat3/cuButterfly/issues) 报告。性能结果必须同时给出
硬件身份、源码和库 hash、算子语义、时间边界、候选选择来源、正确性记录和原始数据路径；缺少
这些元数据的读数只能作为开发诊断，不能成为论文 headline。
