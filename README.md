# 基于深度哈希编码的短视频音乐相似性计算

工程框架

``` text
├── bases						# 基类
│   ├── data_loader_base.py 	
│   ├── infer_base.py
│   ├── model_base.py
│   └── trainer_base.py
├── configs 					# 配置
│   └── triplet_config.json 		# 配置类（核心）
├── data_loaders 				# 数据
│   ├── data_augment.py 			# 数据增强
│   ├── data_downloader.py 			# 数据下载
│   ├── data_merger.py 				# 数据组合
│   ├── data_spliter.py 			# 数据分离
│   └── triplet_dl.py 				# 读取类（核心）
├── experiments 				# 实验
├── hash 						# Hash
│   ├── distance_api.py 			# Hash接口（核心）
│   └── hash_preprocessor.py 		# 数据转Hash
├── infers 						# 测试
│   └── triplet_infer.py 			# 测试类（核心）
├── main_predict.py 			# 入口 - 预测
├── main_test.py 				# 入口 - 测试
├── main_train.py 				# 入口 - 训练
├── models 						# 模型
│   └── triplet_model.py 			# 模型类（核心）
├── pyAudioAnalysis 			# 特征提取
├── requirements-gpu.txt
├── requirements.txt
├── root_dir.py
├── trainers 					# 训练
│   └── triplet_trainer.py  		# 训练类
└── utils
    ├── config_utils.py 		# 配置工具类
    ├── np_utils.py 			# NP工具类
    └── utils.py 				# 其他工具类
```

By C. L. Wang @ [美图](http://www.meipai.com/)云事业部