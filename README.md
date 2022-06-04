1. Python 3.7 
2.  
```shell
git clone https://github.com/krm-shrftdnv/diploma-mrcnn.git
```

3. В директории проекта
```shell
python -m venv путь/до/директории/проекта/venv
```
4. 
```shell
pip install -r requirements.txt
```
tensorflow - v1.15.0
keras - v2.3.4
tensorflow-gpu - v.1.15.0
5. 
```shell
python src/count_gpu.py
```
Если есть предупреждения и не видит GPU - фиксим по варнингам:

[Installing cuDNN on Windows](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-10.0-download-archive)
- [Zlib binaries](http://gnuwin32.sourceforge.net/downlinks/zlib-bin-zip.php)
6. Заменить mask_rcnn_coco.h5.tmp на [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
7. Если всё ок:
```shell
python src/train.py
```
7. После обучения в _src/inference.py_ поменять MODEL_PATH на путь до последнего созданного **.h5** файла в директории _models/train*_
8. Тестим
```shell
python src/inference.py
```