

# git clone https://github.com/CUHK-AIM-Group/U-KAN.git
# cd U-KAN
# conda create -n ukan python=3.10
# conda activate ukan
# cd Seg_UKAN && pip install -r requirements.txt

# cd Seg_UKAN
#python train.py --arch UKAN --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_UKAN  --data_dir [YOUR_DATA_DIR]

python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_UKAN  --data_dir ./inputs