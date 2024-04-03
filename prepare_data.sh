dataroot1=./data
mkdir ${dataroot1}
aws s3 cp s3://ec2-sd-images/updated-VITON-HD ${dataroot1}

unzip ./data/updated-VITON-HD -d ${dataroot1}
rm -rf ./data/updated-VITON-HD
