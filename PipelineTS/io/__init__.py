import datetime
from copy import deepcopy


def _get_object_name(model_obj):
    """
    Extracts and returns the object name from its string representation.

    Parameters
    ----------
    model_obj : object
        The object for which the name needs to be extracted.

    Returns
    -------
    str
        The extracted object name.
    """
    import re
    return re.split('<|>', str(model_obj))[1].split(' ')[0].strip().split('.')[-1]


def _hash_string(string):
    """
    Hashes a given string using MD5 algorithm.

    Parameters
    ----------
    string : str
        The input string to be hashed.

    Returns
    -------
    str
        The first 12 characters of the hashed string.
    """
    import hashlib
    import random
    import time

    # Create an MD5 hash object
    hash_object = hashlib.md5()

    # Update the hash object's value with the string to be encrypted
    hash_object.update((string + str(random.random()) + str(time.time())).encode('utf-8'))

    # Get the encrypted result
    encrypted_string = hash_object.hexdigest()

    return encrypted_string[:12]


def _zip_file(zipfile_fp, *file_fp):
    """
    Compresses multiple files into a zip archive.

    Parameters
    ----------
    zipfile_fp : str
        The path to the zip file.

    *file_fp : str
        Variable length list of file paths to be compressed.
    """
    import zipfile
    from pathlib import Path

    from spinesUtils.asserts import raise_if_not

    raise_if_not(ValueError, zipfile_fp.endswith('.zip'), "`zipfile_fp` must be a string with the `.zip` suffix")

    with zipfile.ZipFile(zipfile_fp, 'w') as zipf:
        for file in file_fp:
            zipf.write(file, Path(file).name)


def _load_zip_file(zipfile_fp):
    """
    Extracts the contents of a zip file to a temporary directory.

    Parameters
    ----------
    zipfile_fp : str
        The path to the zip file.

    Returns
    -------
    str
        The path to the temporary directory containing the extracted contents.
    """
    import zipfile
    from pathlib import Path

    from spinesUtils.asserts import raise_if, raise_if_not
    raise_if(ValueError, Path(zipfile_fp).is_dir(), "`zipfile_fp` must be a file name, not a directory.")

    raise_if_not(ValueError, zipfile_fp.strip().endswith('.zip'),
                 "`zipfile_fp` must be a string with the `.zip` suffix")

    # To ensure that folder names are as unique as possible
    tmp_unzip_fp = str(Path(zipfile_fp).parent.absolute().joinpath(
        f'PIPELINETS_MODEL_{_hash_string(zipfile_fp)}_{int(datetime.datetime.now().timestamp()*1e6)}/'))

    with zipfile.ZipFile(zipfile_fp, 'r') as zip_ref:
        zip_ref.extractall(tmp_unzip_fp)

    return tmp_unzip_fp


def _save_single_model(path, model, scaler=None):
    """
    Save a machine learning model along with its scaler to a zip file.

    Parameters
    ----------
    path : str
        The path to the zip file.

    model : object
        The machine learning model to be saved.

    scaler : object, optional
        The scaler associated with the model.

    Returns
    -------
    str
        The path to the saved zip file.
    """
    from pathlib import Path
    import shutil

    import cloudpickle
    from spinesUtils.asserts import raise_if, raise_if_not

    raise_if(ValueError, Path(path).is_dir(), "`path` must be a file name, not a directory.")
    raise_if_not(ValueError, path.endswith('.zip'), "`path` must be a string with the `.zip` suffix")

    path = str(Path(path).absolute())

    zipfile_fp = path

    model_fp = Path(path.strip()[:-4] + '/')

    # Create model directory
    Path.mkdir(model_fp)

    pkl_file_fp = str(model_fp.joinpath(Path(path.strip()[:-4] + '.pkl').name))

    with open(pkl_file_fp, 'wb') as f:
        if scaler is not None:
            cloudpickle.dump([model, scaler], f)
        else:
            cloudpickle.dump(model, f)

    _zip_file(zipfile_fp, pkl_file_fp)

    shutil.rmtree(model_fp)

    return zipfile_fp


def _load_single_model(path, unzip_file_path=None, unzip=True):
    """
    Load a machine learning model from a zip file.

    Parameters
    ----------
    path : str
        The path to the zip file.

    unzip_file_path : str, optional
        The path to a pre-extracted directory containing the contents of the zip file.

    unzip : bool, optional
        If True, the zip file is extracted to a temporary directory.

    Returns
    -------
    tuple or object
        If a scaler is saved along with the model, returns a tuple (model, scaler). Otherwise, returns the model.
    """
    import os
    from pathlib import Path
    import shutil

    import cloudpickle
    from spinesUtils.asserts import raise_if, raise_if_not

    raise_if(ValueError, Path(path).is_dir(), "`path` must be a file name, not a directory.")
    raise_if_not(ValueError, path.endswith('.zip'), "`path` must be a string with the `.zip` suffix")
    raise_if(ValueError, unzip_file_path is None and unzip is False,
             "`unzip_file_path` must be specified when `unzip` is False.")
    raise_if(ValueError, unzip_file_path is not None and unzip is True,
             "`unzip_file_path` must be None when `unzip` is True.")

    if unzip:
        unzip_file_fp = _load_zip_file(path)
    else:
        unzip_file_fp = unzip_file_path

    raise_if_not(ValueError, any([i.endswith('.pkl') for i in os.listdir(unzip_file_fp)]),
                 "Zip file must contain one file with the `.pkl` suffix.")

    model = None
    scaler = None

    for i in os.listdir(unzip_file_fp):
        if i.endswith('.pkl'):
            with open(str(Path(unzip_file_fp).joinpath(i)), 'rb') as f:
                model = cloudpickle.load(f)

            if isinstance(model, list) and len(model) == 2:
                (model, scaler) = model

    shutil.rmtree(unzip_file_fp)

    if scaler is not None:
        return model, scaler

    return model


def _save_pipeline(path, model):
    """
    Save a machine learning pipeline to a zip file.

    Parameters
    ----------
    path : str
        The path to the zip file.

    model : object
        The machine learning pipeline to be saved.
    """
    from pathlib import Path
    import shutil

    import cloudpickle
    from spinesUtils.asserts import raise_if, raise_if_not

    raise_if(ValueError, Path(path).is_dir(), "`path` must be a file name, not a directory.")
    raise_if_not(ValueError, path.endswith('.zip'), "`path` must be a string with the `.zip` suffix")

    zipfile_fp = path

    # Make a directory with the pkl file name
    name_subfix = _hash_string(path.strip()[:-4])
    pipeline_path = Path(path).parent.joinpath(path.strip()[:-4] + name_subfix + '/')
    Path.mkdir(pipeline_path)

    pkl_file_fp = str(Path(pipeline_path).joinpath('pipeline.pkl'))

    file_fps = [pkl_file_fp]
    for (sub_model_name, sub_model) in model.models_:
        mpath = str(pipeline_path.joinpath(sub_model_name + '.zip'))
        file_fps.append(mpath)
        _save_single_model(mpath, sub_model)

    with open(pkl_file_fp, 'wb') as f:
        new_model = deepcopy(model)
        new_model.models_ = []
        new_model.best_model_ = None
        cloudpickle.dump(new_model, f)

    _zip_file(zipfile_fp, *file_fps)

    shutil.rmtree(pipeline_path)

    return zipfile_fp


def _load_pipeline(path, unzip_file_path=None, unzip=True):
    """
    Load a machine learning pipeline from a zip file.

    Parameters
    ----------
    path : str
        The path to the zip file.

    unzip_file_path : str, optional
        The path to a pre-extracted directory containing the contents of the zip file.

    unzip : bool, optional
        If True, the zip file is extracted to a temporary directory.

    Returns
    -------
    object
        The loaded machine learning pipeline.
    """
    import os
    from pathlib import Path
    import shutil

    import cloudpickle
    from spinesUtils.asserts import raise_if, raise_if_not

    raise_if(ValueError, Path(path).is_dir(), "`path` must be a file name, not a directory.")
    raise_if_not(ValueError, path.endswith('.zip'), "`path` must be a string with the `.zip` suffix")
    raise_if(ValueError, unzip_file_path is None and unzip is False,
             "`unzip_file_path` must be specified when `unzip` is False.")
    raise_if(ValueError, unzip_file_path is not None and unzip is True,
             "`unzip_file_path` must be None when `unzip` is True.")

    if unzip:
        unzip_file_fp = _load_zip_file(path)
        unzip_file_fp = str(Path(unzip_file_fp))
    else:
        unzip_file_fp = unzip_file_path

    raise_if_not(ValueError, os.listdir(unzip_file_fp).count('pipeline.pkl') == 1,
                 "Zip file must contain one file which be named `pipeline.pkl`")

    pipeline = None
    models = []
    for i in os.listdir(unzip_file_fp):
        if i == 'pipeline.pkl':
            with open(str(Path(unzip_file_fp).joinpath(i)), 'rb') as f:
                pipeline = cloudpickle.load(f)
        else:
            models.append([i[:-4], _load_single_model(str(Path(unzip_file_fp).joinpath(i)))])

    pipeline.models_ = models
    for (sub_model_name, sub_model) in pipeline.models_:
        if sub_model_name == pipeline.leader_board_.iloc[0, :]['model']:
            pipeline.best_model_ = sub_model

    shutil.rmtree(unzip_file_fp)

    return pipeline


def _save_smart_router(path, router):
    """Save a SmartRouter to a zip file.

    Saves the router metadata and the inner pipeline separately.
    """
    from pathlib import Path
    import shutil
    import cloudpickle
    from spinesUtils.asserts import raise_if, raise_if_not

    raise_if(ValueError, Path(path).is_dir(), "`path` must be a file name, not a directory.")
    raise_if_not(ValueError, path.endswith('.zip'), "`path` must be a string with the `.zip` suffix")
    raise_if(ValueError, router.pipeline_ is None, "SmartRouter has not been fitted. Call fit() first.")

    zipfile_fp = path
    name_subfix = _hash_string(path.strip()[:-4])
    router_dir = Path(path).parent.joinpath(path.strip()[:-4] + name_subfix + '/')
    Path.mkdir(router_dir)

    file_fps = []

    # Save inner pipeline as nested zip
    pipeline_zip = str(router_dir.joinpath('inner_pipeline.zip'))
    _save_pipeline(pipeline_zip, router.pipeline_)
    file_fps.append(pipeline_zip)

    # Save router metadata (everything except pipeline_ and logger)
    meta_fp = str(router_dir.joinpath('router_meta.pkl'))
    meta = {
        'time_col': router.time_col,
        'target_col': router.target_col,
        'n_predict': router.n_predict,
        'quantile': router.quantile,
        'accelerator': router.accelerator,
        'random_state': router.random_state,
        'verbose': router.verbose,
        'max_models': router.max_models,
        'cv': router.cv,
        'time_limit': router.time_limit,
        'ensemble_strategy': router.ensemble_strategy,
        'ensemble_top_k': router.ensemble_top_k,
        'search_strategy': router.search_strategy,
        'profile_': router.profile_,
        'strategy_': router.strategy_,
        'leader_board_': router.leader_board_,
        'model_scores_': router.model_scores_,
        'ensemble_': router.ensemble_,
        '_scaler_obj': router._scaler_obj,
        '_screening_results': router._screening_results,
        '_lag_exploration_results': router._lag_exploration_results,
        '_calibration_rho': router._calibration_rho,
    }
    with open(meta_fp, 'wb') as f:
        cloudpickle.dump(meta, f)
    file_fps.append(meta_fp)

    # Marker file so load_model can detect SmartRouter
    marker_fp = str(router_dir.joinpath('smart_router.marker'))
    with open(marker_fp, 'w') as f:
        f.write('SmartRouter')
    file_fps.append(marker_fp)

    _zip_file(zipfile_fp, *file_fps)
    shutil.rmtree(router_dir)
    return zipfile_fp


def _load_smart_router(path, unzip_file_path=None, unzip=True):
    """Load a SmartRouter from a zip file."""
    import os
    from pathlib import Path
    import shutil
    import cloudpickle
    from spinesUtils.asserts import raise_if, raise_if_not

    raise_if(ValueError, Path(path).is_dir(), "`path` must be a file name, not a directory.")
    raise_if_not(ValueError, path.endswith('.zip'), "`path` must be a string with the `.zip` suffix")

    if unzip:
        unzip_file_fp = _load_zip_file(path)
    else:
        unzip_file_fp = unzip_file_path

    # Load metadata
    meta_fp = str(Path(unzip_file_fp).joinpath('router_meta.pkl'))
    with open(meta_fp, 'rb') as f:
        meta = cloudpickle.load(f)

    # Load inner pipeline
    pipeline_zip = str(Path(unzip_file_fp).joinpath('inner_pipeline.zip'))
    pipeline = _load_pipeline(pipeline_zip)

    # Reconstruct SmartRouter
    from PipelineTS.pipeline.smart_router import SmartRouter
    router = SmartRouter(
        time_col=meta['time_col'],
        target_col=meta['target_col'],
        n_predict=meta.get('n_predict'),
        quantile=meta.get('quantile'),
        accelerator=meta.get('accelerator', 'auto'),
        random_state=meta.get('random_state', 0),
        verbose=meta.get('verbose', True),
        max_models=meta.get('max_models', 8),
        cv=meta.get('cv', 5),
        time_limit=meta.get('time_limit'),
        ensemble_strategy=meta.get('ensemble_strategy', 'auto'),
        ensemble_top_k=meta.get('ensemble_top_k', 3),
        search_strategy=meta.get('search_strategy', 'auto'),
    )
    router.pipeline_ = pipeline
    router.profile_ = meta.get('profile_')
    router.strategy_ = meta.get('strategy_')
    router.leader_board_ = meta.get('leader_board_')
    router.best_model_ = pipeline.best_model_
    router.model_scores_ = meta.get('model_scores_')
    router.ensemble_ = meta.get('ensemble_')
    router._scaler_obj = meta.get('_scaler_obj')
    router._screening_results = meta.get('_screening_results')
    router._lag_exploration_results = meta.get('_lag_exploration_results')
    router._calibration_rho = meta.get('_calibration_rho')

    # Re-link ensemble pipeline reference
    if router.ensemble_ is not None:
        router.ensemble_.pipeline = pipeline

    shutil.rmtree(unzip_file_fp)
    return router


def save_model(path, model, scaler=None):
    """
    Save a machine learning model, pipeline, or SmartRouter to a zip file.

    [Note that]: A loaded model cannot be saved.

    Parameters
    ----------
    path : str
        The path to the zip file.

    model : object
        The fitted machine learning model, ModelPipeline, or SmartRouter.

    scaler : object, optional
        The scaler associated with the model (only for single models).

    Returns
    -------
    str
        The path to the saved zip file.
    """
    from PipelineTS.pipeline import ModelPipeline
    from PipelineTS.pipeline.smart_router import SmartRouter
    if isinstance(model, SmartRouter):
        return _save_smart_router(path, model)
    elif isinstance(model, ModelPipeline):
        return _save_pipeline(path, model)
    else:
        return _save_single_model(path, model, scaler)


def load_model(path):
    """
    Load a machine learning model, pipeline, or SmartRouter from a zip file.

    Parameters
    ----------
    path : str
        The path to the zip file.

    Returns
    -------
    object
        The loaded machine learning model, ModelPipeline, or SmartRouter.
    """
    import os
    unzip_file_fp = _load_zip_file(path)

    if 'smart_router.marker' in os.listdir(unzip_file_fp):
        return _load_smart_router(path, unzip_file_path=unzip_file_fp, unzip=False)
    elif 'pipeline.pkl' in os.listdir(unzip_file_fp):
        return _load_pipeline(path, unzip_file_path=unzip_file_fp, unzip=False)
    else:
        return _load_single_model(path, unzip_file_path=unzip_file_fp, unzip=False)
