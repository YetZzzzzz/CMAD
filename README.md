# CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities
The code for CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities, which is accepted in [ICCV 25](https://iccv.thecvf.com/virtual/2025/poster/2166).
### The Framework of CMAD:
![image](https://github.com/YetZzzzzz/CMAD/blob/main/CMAD_framework.png)
Figure: Overview of the proposed CMAD framework. It consists of a student model, a teacher model, and two key modules: Correlation-Aware Feature Distillation (CAFD) and Modalities-Aware Regularization (MAR). CAFD ensures feature matching between student-teacher pairs and correlation alignment across samples between student-teacher and teacher-teacher representations, while MAR dynamically adjusts the weight of each modality combinations based on its difficulty.

### Datasets:
**Please move the following datasets into directory ./datasets/.**

The CMU-MOSEI dataset can be downloaded according to [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) and [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) through the following link: 
```
pip install gdown
gdown https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3
```

For UR-FUNNY and MUStARD, the dataset can be downloaded according to [HKT](https://github.com/matalvepu/HKT/blob/main/dataset/download.txt) through:
```
Download Link of UR-FUNNY: https://www.dropbox.com/s/5y8q52vj3jklwmm/ur_funny.pkl?dl=1
Download Link of MUStARD: https://www.dropbox.com/s/w566pkeo63odcj5/mustard.pkl?dl=1
```
Please rename the files as ur_funny.pkl and mustard.pkl, and move them into the directory ./datasets/.

For CHERMA dataset, you can download from [LFMIM](https://github.com/sunjunaimer/LFMIM) through: 
```
https://pan.baidu.com/s/10PoJcXMDhRg4fzsq96A7rQ
Extraction code: CHER
```
Please put the files into directory ./datasets/CHERMA0723/.

### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions**


### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT-EN/.


