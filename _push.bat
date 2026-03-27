@echo off
cd /d D:\brainwave-morse
git add -A
git commit -m "Update: Fix data leakage, real results 71.5%% LOSO"
git push origin main
echo [DONE]
