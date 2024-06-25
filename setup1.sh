conda remove --name rppg-toolbox --all -y
conda create -n rppg-toolbox -y #python=3.8 pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda activate rppg-toolbox
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121i

#conda install pytorch torchvision==0.14.0 torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
