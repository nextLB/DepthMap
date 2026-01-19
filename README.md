# 深度图检测

## 环境配置

来到Depth文件夹下
    
    cd Depth

创建conda环境以及安装依赖

    conda create -n Depth python=3.11

    conda activate Depth

    pip install -r requirements.txt

验证环境是否配置成功，可以运行如下指令试下看

    python run.py --encoder vitl --img-path assets/examples --outdir depth_vis


若出现对应的depth_vis文件夹及其对应的结果图像即为运行成功


## 关于显存使用的查看

    watch -n 2 nvidia-smi



