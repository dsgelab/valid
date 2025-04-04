import optuna

def create_optuna_study(study_name: str, 
                        file_name: str, 
                        sampler: optuna.samplers,
                        model_fit_date: str,
                        res_dir: str,
                        refit: bool) -> optuna.study.Study:
    """Creates an Optuna study object."""
    store_name = "sqlite:///" + res_dir + file_name + "_" + model_fit_date + "_optuna.db"
    if refit: 
        try:
            optuna.delete_study(study_name=study_name, storage=store_name)
        except KeyError:
            print("Record not found, have to create new anyways.")
    study = optuna.create_study(direction='minimize', 
                                sampler=sampler, 
                                study_name=study_name, 
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                storage=store_name, 
                                load_if_exists=True)
    return(study)