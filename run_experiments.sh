python run.py \
  -data ./datasets \
  -dataset-name stl10 \
  --log-every-n-steps 100 \
  --epochs 2 \
  --arch resnet18 \
  --seed 7 \
  --out_dim 128 \
  --workers 8 \
  --kernel cosine_similarity

python run.py \
  -data ./datasets \
  -dataset-name stl10 \
  --log-every-n-steps 100 \
  --epochs 2 \
  --arch resnet18 \
  --seed 7 \
  --out_dim 128 \
  --workers 8 \
  --gamma1 1 \
  --kernel laplacian

python run.py \
  -data ./datasets \
  -dataset-name stl10 \
  --log-every-n-steps 100 \
  --epochs 2 \
  --arch resnet18 \
  --seed 7 \
  --out_dim 128 \
  --workers 8 \
  --gamma1 0.5 \
  --kernel exponential

python run.py \
  -data ./datasets \
  -dataset-name stl10 \
  --log-every-n-steps 100 \
  --epochs 2 \
  --arch resnet18 \
  --seed 7 \
  --out_dim 128 \
  --workers 8 \
  --gamma1 1 \
  --gamma2 2 \
  --gamma_lambda 0.5 \
  --kernel simple




#positives.shape=torch.Size([512, 1])
#negatives.shape=torch.Size([512, 510])
