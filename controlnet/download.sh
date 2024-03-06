# aws s3 cp s3://ec2-sd-images/checkpoints/vgg19_conv.pth ./models/vgg/
# aws s3 cp s3://ec2-sd-images/checkpoints/warp_viton.pth ./models/checkpoints/
# aws s3 cp s3://ec2-sd-images/checkpoints/viton512.ckpt ./models/checkpoints/

dataroot1=./data
mkdir ${dataroot1}
aws s3 cp s3://ec2-sd-images/updated-VITON-HD ${dataroot1}

unzip ./data/updated-VITON-HD -d ${dataroot1}
rm -rf ./data/updated-VITON-HD