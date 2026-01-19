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

同时可以测试一下相关app是否可以使用，可以运行如下指令试下看:

    python app.py

若出现报错，或许是hugging face的版本不正确

重新安装一下hugging face,指定对应的版本

    pip install huggingface_hub==0.25.2

再次运行app.py,然后复制终端出现的网址在本地浏览器，进行访问即可
    




## 关于显存使用的查看

    watch -n 2 nvidia-smi



