运行指南
==================
1.部署kaldi的运行环境<br>
2.部署tensorflow的运行环境<br>
3.利用kaldi的数据准备脚本，准备数据<br>
4.修改conf/my.conf中data和sample里的参数,运行make_fbank.py生成特征文件<br>
5.修改conf/my.conf中相关参数，运行train.py进行模型训练<br>
6.根据log中的dev_accuracy_log参数指定的文件所显示的结果较好的steps来修改conf/my.conf中produce中的check_point参数，然后指定生成dvector的feats.scp的路径，修改其他相关参数，运行produce_dvector.py生成对应dvector<br>
7.运行compute_eer.py计算eer，此处采用的是余弦距离<br>
8.采用kaldi的脚本来进行测试，先source加载kaldi的运行环境，生成用来做LDA的spk2utt,然后运行tool/train_lda_plda.sh训练plda模型<br>
9.运行tool/eer_all_score.sh来计算plda的eer，其中需要指定kaldi例子中local的路径，以及是否生成新的trials，如果第一次运行选择输入yes<br>
