<div align="center">
<h1>CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception</h1>

[Jiaru Zhong](https://scholar.google.com/citations?hl=zh-CN&user=Q9KMoxkAAAAJ), Jiahao Wang, [Jiahui Xu](https://scholar.google.com/citations?hl=zh-CN&user=MHa9ts4AAAAJ), [Xiaofan Li](https://scholar.google.com/citations?hl=zh-CN&user=pjZdkO4AAAAJ&view_op=list_works&sortby=pubdate), [Zaiqing Nie](https://scholar.google.com/citations?user=Qg7T6vUAAAAJ), [Haibao Yu](https://scholar.google.com/citations?user=JW4F5HoAAAAJ)\*</sup>

<!-- <sup>1</sup> The Hong Kong Polytechnic University <sup>2</sup> AIR, Tsinghua University <br> <sup>3</sup> The University of Hong Kong <sup>4</sup> SVM, Tsinghua University <br> <sup>5</sup> Baidu Inc.
<br> Work done while at AIR, Tsinghua University. -->

<!-- ![CoopTrack](https://img.shields.io/badge/Arxiv-Paper-2b9348.svg?logo=arXiv)(Coming Soon)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Download-blue)]()&nbsp; -->

</div>

## News

- **` June 26, 2025`:** CoopTrack has been accepted by ICCV 2025! We will release our paper and code soon!

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started) (Coming Soon)
- [Contact](#contact)
- [Citation](#citation) (Coming Soon)
- [Related Works](#related-works)

## Introduction
Cooperative perception aims to address the inherent limitations of single autonomous driving systems through information exchange among multiple agents. Previous research has primarily focused on single-frame perception tasks. However, the more challenging cooperative sequential perception tasks, such as cooperative 3D multi-object tracking, have not been thoroughly investigated.
Therefore, we propose CoopTrack, a fully instance-level end-to-end framework for cooperative tracking, featuring learnable instance association, which fundamentally differs from existing approaches. CoopTrack transmits sparse instance-level features that significantly enhance perception capabilities while maintaining low transmission costs. Furthermore, the framework comprises three key components: Multi-Dimensional Feature Extraction (MDFE), Cross-Agent Alignment (CAA), and Graph-Based Association (GBA), which collectively enable comprehensive instance representation with semantic and motion features, and adaptive cross-agent association and fusion based on graph learning. Experiments on the V2X-Seq dataset demonstrate that, benefiting from its sophisticated design, CoopTrack achieves state-of-the-art performance, with 39.0% mAP and 32.8% AMOTA.


## Getting Started
We will release codes soon. Stay tuned!


## Contact

If you have any questions, please contact Jiaru Zhong via email (zhong.jiaru@outlook.com).

<!-- ## Acknowledgement

This work is partly built upon [UniV2X](https://github.com/AIR-THU/UniV2X), [UniAD](https://github.com/OpenDriveLab/UniAD), [PF-Track](https://github.com/TRI-ML/PF-Track), and [AdaTrack](https://github.com/dsx0511/ADA-Track). Thanks them for their great works! -->

## Citation
We will release our paper on arXiv soon.
<!-- If you find CoopTrack is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex

``` -->

## Related Works
- [UniV2X](https://github.com/AIR-THU/UniV2X)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
- [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq)
- [PF-Track](https://github.com/TRI-ML/PF-Track)
- [AdaTrack](https://github.com/dsx0511/ADA-Track)
- [FFNET](https://github.com/haibao-yu/FFNet-VIC3D)