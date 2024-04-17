# VocaloidAITuner
vocaloid ai调校，
## 推理 inference
```
python main.py --i your_input_file.vsqx  --o output_file.vsqx
```
## 训练 training
把vsqx文件放在vcfiles里，
```
python preprocess.py
```
自动命名并生成trainfiles, testfiles
```
python train.py
```
模型会保存在 './model.pth'
