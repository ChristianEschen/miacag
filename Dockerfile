FROM nvcr.io/nvidia/pytorch:21.12-py3
ARG DEBIAN_FRONTEND=noninteractive
COPY requirements.txt  /tmp/

# postgres
RUN apt-get update -y && \ 
   apt-get install tk-dev -y && \
   rm -r /var/lib/apt/lists/* && \
   apt-get update -y && \ 
   apt-get install libpq-dev -y

# pip packages
RUN pip install --requirement /tmp/requirements.txt

# download pretrained models
RUN `wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth`
RUN `wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth`
RUN `wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_32x3_f294077834.pyth`
RUN `wget https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth`
# move pretrained models to miacag
RUN miapath=`python -c "import miacag, os; print(os.path.dirname(miacag.__file__))"` && \
    mv "X3D_S.pyth" "$miapath/models/torchhub/2D+T/x3d_s/model.pt" && \
    mv "MVIT_B_16x4.pyth" "$miapath/models/torchhub/2D+T/mvit_base_16x4/model.pt" && \
    mv "MVIT_B_32x3_f294077834.pyth" "$miapath/models/torchhub/2D+T/mvit_base_32x3/model.pt" && \
    mv "r2plus1d_18-91a641e6.pth" "$miapath/models/torchhub/2D+T/r2plus1_18/model.pt"