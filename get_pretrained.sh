wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth
wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth
wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_32x3_f294077834.pyth
wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth
# move pretrained models to miacag
miapath=`python -c "import miacag, os; print(os.path.dirname(miacag.__file__))"`
mv "X3D_S.pyth" "$miapath/models/torchhub/2D+T/x3d_s/model.pt"
mv "MVIT_B_16x4.pyth" "$miapath/models/torchhub/2D+T/mvit_base_16x4/model.pt"
mv "MVIT_B_32x3_f294077834.pyth" "$miapath/models/torchhub/2D+T/mvit_base_32x3/model.pt"
mv "R2PLUS1D_16x4_R50.pyth" "$miapath/models/torchhub/2D+T/r2plus1_18/model.pt"