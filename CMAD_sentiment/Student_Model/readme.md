You can train the teacher model from scratch using scripts in ```/CMAD/CMAD_sentiment/Teacher_Model```, you can also try to download the [checkpoint](https://pan.baidu.com/s/1bioKGv393xl7JjYXHGf8Ow?pwd=px6s) (Extraction code: px6s) we provided and put in directory ```/CMAD/CMAD_sentiment/Teacher_Model```.

For MOSEI dataset, please run the following code to train the **student model**:
```
python3 stu_config_mosei.py --dataset='mosei' --begin_epoch=20 --d_l=96 --delta=0.1 --depth=5 --gamma=1 --latent_dim=96 --learning_rate=2e-5 --n_epochs=80 --tau=0.2 --te_layers=2 --temperature=7.0 --train_batch_size=128
```
