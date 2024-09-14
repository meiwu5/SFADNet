## SFADNet: Spatio-temporal Fused Graph based on Attention Decoupling Network for Traffic Prediction
![model](https://github.com/user-attachments/assets/4cb52d33-7ef6-47c2-8c17-5df6fa640899)
### Abstract
In recent years, traffic flow prediction has become essential for managing intelligent transportation systems, yet traditional methods often fall short due to their reliance on static spatial modeling, which hampers their ability to accurately capture the intricate dynamic relationships between time and space. This paper introduces SFADNet, a novel traffic flow prediction network that categorizes traffic flow into multiple patterns using temporal and spatial feature matrices. For each identified pattern, SFADNet constructs an independent adaptive spatio-temporal fusion graph through a cross-attention mechanism, integrating residual graph convolution modules and time series components to effectively model dynamic spatio-temporal relationships across various fine-grained traffic patterns. Extensive experiments across four large-scale datasets show that SFADNet significantly outperforms current state-of-the-art baselines, highlighting its effectiveness in traffic flow prediction.
### Datasets
The dataset can be downloaded from here: 
### Run
Place the downloaded dataset in the datasets directory, and then configure the environment. 
\\
"conda create -n SFADNet python==3.8"
\\
"conda activate SFADNet"
\\
"cd SFADNet"
\\
"pip install -r requirement.txt"
