# 
file=test/a1_test_new_args.py
nohup python $file \
    --train-data 'aa' \
    > $file.log 2>&1 &