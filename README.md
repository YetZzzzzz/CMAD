# CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities
The code for CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities, which is accepted in [ICCV 25](https://iccv.thecvf.com/virtual/2025/poster/2166).
### The Framework of CMAD:
![image](https://github.com/YetZzzzzz/CMAD/blob/main/CMAD_framework.png)
Figure: Overview of the proposed CMAD framework. It consists of a student model, a teacher model, and two key modules: Correlation-Aware Feature Distillation (CAFD) and Modalities-Aware Regularization (MAR). CAFD ensures feature matching between student-teacher pairs and correlation alignment across samples between student-teacher and teacher-teacher representations, while MAR dynamically adjusts the weight of each modality combinations based on its difficulty.

### Datasets:
**Please move the following datasets into directory ```./datasets/```.**

The unaligned CMU-MOSEI dataset can be downloaded according to [DiCMoR](https://github.com/mdswyz/DiCMoR) and [IMDer](https://github.com/mdswyz/IMDer), rename the pkl as ```mosei.pkl```. 


The IEMOCAP dataset can be downloaded according to [MuLT](https://github.com/yaohungt/Multimodal-Transformer/tree/master).

Please put the files into directory ```./datasets/```.

### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions.**

### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ```./BERT-EN/```.

### Run CMAD
You can train the teacher model from scratch using scripts in ```/CMAD/CMAD_sentiment/Teacher_Model```, you can also try to download the [checkpoint](https://pan.baidu.com/s/1bioKGv393xl7JjYXHGf8Ow?pwd=px6s) (Extraction code: px6s) we provided and put in directory ```/CMAD/CMAD_sentiment/Teacher_Model```.

For MOSEI dataset, please run the following code in ```/CMAD/CMAD_sentiment/Student_Model```:
```
python3 stu_config_mosei.py --dataset='mosei' --begin_epoch=20 --d_l=96 --delta=0.1 --depth=5 --gamma=1 --latent_dim=96 --learning_rate=2e-5 --n_epochs=80 --tau=0.2 --te_layers=2 --temperature=7.0 --train_batch_size=128
```

### Citation:
Please cite our paper if you find our work useful for your research:
```
@inproceedings{zhuang2025cmad,
  title={CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities},
  author={Zhuang, Yan and Liu, Minhao and Bai, Wei and Zhang, Yanru and Zhang, Xiaoyue and Deng, Jiawen and Ren, Fuji},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4626--4636},
  year={2025}
}
```

### Acknowledgement
Thanks to  [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) , [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer),  [DiCMoR](https://github.com/mdswyz/DiCMoR), [IMDer](https://github.com/mdswyz/IMDer), [GCNet](https://github.com/zeroQiaoba/GCNet), [LNLN](https://github.com/Haoyu-ha/LNLN), [HKT](https://github.com/matalvepu/HKT), [LFMIM](https://github.com/sunjunaimer/LFMIM) and [MMANet](https://github.com/shicaiwei123/MMANet-CVPR2023/tree/main) for their great help to our codes and research.
