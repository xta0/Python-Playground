pip install -U datasets==2.17.0

pip install --upgrade pip
pip install --disable-pip-version-check \
 torch==1.13.1 \
 torchdata==0.5.1 --quiet

pip install \
 transformers==4.27.2 \
 evaluate==0.4.0 \
 rouge_score==0.1.2 \
 loralib==0.1.1 \
 peft==0.3.0 --quiet
