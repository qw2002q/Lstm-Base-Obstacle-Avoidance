【主要文件】  
ctrl_node.py: 运行小车模块，需要放置与ros节点中，并使用ros运行  
test_drive.py: 使用python，用于模拟测试ctrl_node.py的功能(无法应用于小车实际控制)  
train_model.py: 训练模型模块  
test_model.py: 测试模型模块(包含可视化数据)  
  
【运行命令样例】  
rosrun ctrl ctrl_node.py params1=xxx params2=xxx【运行小车控制模块】  
python train_model.py/test_model.py params1=xxx params2=xxx 【训练/测试模型】  
  
【可用参数】  
train_model.py: 用于训练模型  
[可用参数]  
model_type=lstm/cnn 【选择模型类型】  
save_path【用于保存模型的文件名（不含文件夹）】  
load_path【用于加载模型的文件名（不含文件夹）】  
batch_size【训练的batch size，默认为16】  
n_epochs【训练的迭代次数】  
seq_length【lstm模型的序列长度，默认为1】  
num_layers【lstm模型的层数，默认为2】  
device=cuda:x/cpu 【选择训练所用的显卡或CPU，如cuda:0】  
lr【学习率，默认0.001】  
dataloader_path 【数据集所在文件夹名(Dataloader文件夹内)】  
pretrained_model=true/false 【加载的模型是否为预训练模型，默认false】  
  
test_model.py：用于测试模型  
[可用参数]  
load_path【用于加载模型的文件名（不含文件夹）】  
view=true/false 【True or False：是否进行视频模拟(测试模型可视化)】  
start 【从测试集第几张图像开始测试】  
end 【从测试集第几张图像结束测试】  
batch_size【测试的batch size，默认为1】  
device=cuda:x/cpu 【选择测试所用的显卡或CPU，如cuda:0】  
dataloader_path 【数据集所在文件夹名(Dataloader文件夹内)】  
  
ctrl_node.py: 用于控制小车  
[按键说明]  
方向键：控制小车前进、后退、转弯  
LCtrl: 减小最高速度  
LShift: 增加最高速度  
R：开始记录数据  
ESC：结束程序  
[可用参数]  
load_path【用于加载模型的文件名（不含文件夹）】  
device=cuda:x/cpu 【选择运行模型所用的显卡或CPU，如cuda:0】  
