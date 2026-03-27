@echo off
cd /d D:\brainwave-morse
git add -A
git commit -m "Update: EEGNet pretraining achieves 77.8%% LOSO - BCICIV 2b results"
git push origin main
echo [DONE]
