import optuna

def create_optuna_study(study_name: str, 
                        file_name: str, 
                        sampler: optuna.samplers, 
                        refit: bool) -> optuna.study.Study:
    """Creates an Optuna study object."""
    if refit: 
        try:
            optuna.delete_study(study_name=study_name, storage="sqlite:///" + file_name + "_optuna.db")
        except KeyError:
            print("Record not found, have to create new anyways.")
    study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name, storage="sqlite:///" + file_name + "_optuna.db", load_if_exists=True)
    return(study)