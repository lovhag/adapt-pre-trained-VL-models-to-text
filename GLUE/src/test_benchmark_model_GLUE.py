from benchmark_model_GLUE import benchmark_on_GLUE_task

benchmark_on_GLUE_task(
        model_name="clipbert-no-visual-features", #"lxmert-no-visual-features"
        model_path=None, #"unc-nlp/lxmert-base-uncased",
        model_weights_path="../../models/data/model-weights/clip-bert/mp_rank_00_model_states.pt", #None
        tokenizer_name="bert-base-uncased",
        task_name="cola", #"rte", 
        train_batch_size=2, 
        eval_batch_size=4, 
        epochs=1, 
        lr=3e-5,
        weight_decay=0.1,
        max_train_samples=None,
        tb_logdir=None,
        dataloader_num_workers=0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        visual_features_path=None,
        visual_boxes_path=None
    )