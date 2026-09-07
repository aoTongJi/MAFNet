ICMR26  🚀 MAFNet🚀 
MAFNet: Multi-frequency-Adaptive-Fusion-Network-for-Real-time-Stereo-Matching

# Abstract
Existing stereo matching networks typically rely on either cost-volume construction based on 3D convolutions or deformation methods based on iterative optimization. The former incurs significant computational overhead during cost aggregation, whereas the latter often lacks the ability to model non-local contextual information. These methods exhibit poor compatibility on resource-constrained mobile devices, limiting their deployment in real-time applications. To address this, we propose a Multi-frequency Adaptive Fusion Network (MAFNet), which can produce high-quality disparity maps using only efficient 2D convolutions. Specifically, we design an adaptive frequency-domain filtering attention module that decomposes the full cost volume into high-frequency and low-frequency volumes. Subsequently, we introduce a Linformer-based low-rank attention to adaptively aggregation high- and low-frequency information, yielding more robust disparity estimation. Extensive experiments demonstrate that the proposed MAFNet significantly outperforms existing real-time methods on public datasets such as Scene Flow and KITTI 2015, showing a favorable balance between accuracy and real-time performance. 
![Example of reconstructions](assets/overall.png)


## Pretrained Models

| Model | Download | Extraction Code |
|---|---|
| [Baidu Netdisk](https://pan.baidu.com/s/1oSQLu-nohAbim3Tccb6yfg) | `qe8e` |



## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex

@inproceedings{xu2026mafnet,
  title={MAFNet: Multi-frequency Adaptive Fusion Network for Real-time Stereo Matching},
  author={Xu, Ao and Zhao, Rujin and Xu, Xiong and Huang, Boceng and Jia, Yujia and Long, Hongfeng and Chen, Fuxuan and Cao, Zilong and Chen, Fangyuan},
  booktitle={Proceedings of the 2026 International Conference on Multimedia Retrieval},
  pages={567--576},
  year={2026}
}

```
