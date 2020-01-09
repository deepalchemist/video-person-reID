# Mars
#python main.py -d mars --height 224 --width 112 --arch=resnet50ts --conv5-stride 2 --batch-norm \
# --print-freq 20 --max-epoch 800 --stepsize 200 --eval-step 50 --train-batch 32 --test-batch 32 \
#--ckpt-dir Mr+Bsl+CsTr+Cr224 --gpu-devices 1 \
#--resume '/home/caffe/code/videoReID/ckpt/old/mr_bsl/best_model.pth.tar' \
#--evaluate

python main.py -d mars \
--seq-len 8 --height 224 --width 112 \
--train-sample-method interval --test-sample-method evenly \
--train-batch 32 --test-batch 32 \
--arch=resnet50ts --conv5-stride 1 --batch-norm --pool-type avg \
--stm cstm cmm \
--max-epoch 150 --stepsize 50 --eval-step 25 --print-freq 100 \
--ckpt-dir Mr+CstmCmmBn14x7+CsTr+pkS8Itv \
--gpu-devices 4,5 \
#--resume '/home/caffe/code/videoReID/ckpt/2020-01-06/Mr+BslBn16x8+CsTr+pk/best_model.pth.tar' \
#--evaluate

#------------------------------------------------------------------
# DukeMTMC-VideoReID

#python main.py -d duke \
#--arch=resnet50tp \
#--print-freq 25 \
#--max-epoch 180 --stepsize 60 --eval-step 60 \
#--train-batch 32 \
#--test-batch 32 \
#--ckpt-dir du_tp_evenly \
#--gpu-devices 1