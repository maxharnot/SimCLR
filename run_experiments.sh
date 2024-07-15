python run.py \
  -data ./datasets \
  -dataset-name stl10 \
  --log-every-n-steps 100 \
  --epochs 2 \
  --arch resnet18 \
  --seed 7 \
  --out_dim 128 \
  --kernel laplacian


#positives.shape=torch.Size([512, 1])
#negatives.shape=torch.Size([512, 510])
