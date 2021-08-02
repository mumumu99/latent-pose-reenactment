for i in $(seq -f "%05g.jpg" 1655 5483)
do
rm /content/drive/MyDrive/latent-pose-reenactment/dataset/images-cropped/driving/$i
done