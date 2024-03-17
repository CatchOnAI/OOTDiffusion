!/bin/bash
apt update
apt-get --assume-yes install unzip

# yum remove awscli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/workspace/awscliv2.zip"
unzip /workspace/awscliv2.zip -d /workspace/
/workspace/aws/install
rm -rf /workspace/awscliv2.zip