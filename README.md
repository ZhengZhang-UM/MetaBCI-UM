这是UM参加Metabci比赛的项目，代码主要实现了基于Metabci的人机交互。
在人与大模型沟通的过程中，采集到EEG信号被解码分类后从后台发送到LLM的显示框中，LLM会根据用户的输入读取语义信息，结合用户手动提交的情感状态，对我们的解码模型进行校准并进行测试时自适应。
新增离线测试主要有两个指标，设置好seediv数据集的路径并运行metabci.brainflow.datasets.seediv.py之后可以看到seediv数据集被成功加载。
另外，打开TTA_offline.py文件并设置好seed数据集的路径之后，可以看到使用测试时自适应方法后的准确率。
This is UM's project for the Metabolic competition. 
The code mainly implements human-computer interaction based on Metabolic. 
During the communication between the user and the large model, the collected EEG signals are decoded and classified, then sent from the backend to the LLM's display window. 
The LLM reads semantic information based on the user's input, combines it with the user's manually submitted emotional state, calibrates our decoding model, and performs adaptive testing.
The newly added offline test mainly has two metrics: running metabci.brainflow.datasets.seediv.py .Afterwards, you can see that the seediv dataset has been successfully loaded.
Additionally, after opening the TTA_offline.py file and setting the path to the seed dataset, you can see the accuracy after using the adaptive method during testing.
