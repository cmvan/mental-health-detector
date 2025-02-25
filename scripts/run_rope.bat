@echo off

@REM Pretrain the model
python src\run.py pretrain rope vocab.txt ^
        --writing_params_path rope.pretrain.params
        
@REM @REM Finetune the model
python src\run.py finetune rope vocab.txt ^
        --reading_params_path rope.pretrain.params ^
        --writing_params_path rope.finetune.params ^
        --finetune_corpus_path crisis_train.tsv
        
@REM Evaluate on the dev set; write to disk
python src\run.py evaluate rope vocab.txt  ^
        --reading_params_path rope.finetune.params ^
        --eval_corpus_path crisis_dev.tsv ^
        --outputs_path rope.pretrain.dev.predictions
        
@REM Evaluate on the test set; write to disk
python src\run.py evaluate rope vocab.txt  ^
        --reading_params_path rope.finetune.params ^
        --eval_corpus_path crisis_test.tsv ^
        --outputs_path rope.pretrain.test.predictions
