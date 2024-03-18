!/bin/bash
sudo apt update
sudo apt-get --assume-yes install unzip

# yum remove awscli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "./awscliv2.zip"
unzip ./awscliv2.zip -d .
./aws/install
rm -rf ./awscliv2.zip