conda remove --name rppg-toolbox --all -y
conda create -n rppg-toolbox -y #python=3.8 pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install pytorch torchvision==0.14.0 torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y