"""
PipelineTS Custom Binary Format (.pts)
=======================================

A self-contained binary format for saving and loading PipelineTS models,
pipelines, and SmartRouters with built-in integrity verification.

File Layout
-----------
┌──────────────────────────────────┐
│ Magic Number       (8 bytes)     │  b'PPTS\\x00\\x01\\x00\\x00'
├──────────────────────────────────┤
│ Format Version     (2 bytes)     │  uint16 LE
├──────────────────────────────────┤
│ Header Length       (4 bytes)    │  uint32 LE
├──────────────────────────────────┤
│ Header (JSON bytes, variable)    │  model_type, sections[], metadata
├──────────────────────────────────┤
│ Section 1 data     (variable)    │  cloudpickle bytes
├──────────────────────────────────┤
│ Section N data     (variable)    │  ...
├──────────────────────────────────┤
│ Global SHA-256     (32 bytes)    │  over all bytes above
├──────────────────────────────────┤
│ Footer Magic       (4 bytes)     │  b'PTSE'
└──────────────────────────────────┘

Security features:
- SHA-256 global checksum over the entire payload (magic → last section byte)
- Per-section SHA-256 checksums stored in the header
- Magic number validation for quick file identification
- Format version for forward/backward compatibility
- Backward compatible: load_model() still accepts legacy .zip files (read-only)
"""

import datetime
import hashlib
import json
import struct
from copy import deepcopy
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

MAGIC = b'PPTS\x00\x01\x00\x00'   # 8 bytes: PipelineTS format identifier
FOOTER_MAGIC = b'PTSE'             # 4 bytes: PipelineTS End marker
FORMAT_VERSION = 1                 # uint16
FILE_EXTENSION = '.pts'

# Model type identifiers stored in header
MODEL_TYPE_SINGLE = 'single_model'
MODEL_TYPE_PIPELINE = 'pipeline'
MODEL_TYPE_SMART_ROUTER = 'smart_router'


# ─── Low-level binary helpers ─────────────────────────────────────────────────

def _sha256(data):
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_bytes(data):
    """Compute SHA-256 raw digest (32 bytes) of bytes."""
    return hashlib.sha256(data).digest()


def _serialize_obj(obj):
    """Serialize a Python object to bytes using cloudpickle."""
    import cloudpickle
    import io
    buf = io.BytesIO()
    cloudpickle.dump(obj, buf, protocol=5)
    return buf.getvalue()


def _deserialize_obj(data):
    """Deserialize bytes back to a Python object using cloudpickle."""
    import cloudpickle
    import io
    return cloudpickle.load(io.BytesIO(data))


def _validate_path(path, for_save=True):
    """Validate file path for save/load operations.

    Parameters
    ----------
    path : str
        File path to validate.
    for_save : bool
        If True, validates for save (only .pts allowed).
        If False, validates for load (.pts or legacy .zip).

    Raises
    ------
    ValueError
        If the path is invalid.
    """
    p = Path(path)
    if p.is_dir():
        raise ValueError(f"`path` must be a file name, not a directory: {path}")
    if for_save:
        if not path.endswith(FILE_EXTENSION):
            raise ValueError(
                f"`path` must end with '{FILE_EXTENSION}'. Got: {path}"
            )
    else:
        if not path.endswith(FILE_EXTENSION) and not path.endswith('.zip'):
            raise ValueError(
                f"`path` must end with '{FILE_EXTENSION}' or '.zip' (legacy). Got: {path}"
            )
    if not for_save and not p.exists():
        raise ValueError(f"File does not exist: {path}")


# ─── Core binary write / read ────────────────────────────────────────────────

def _write_pts(path, model_type, sections, metadata=None):
    """Write a .pts binary file.

    Parameters
    ----------
    path : str
        Output file path (must end with .pts).
    model_type : str
        One of MODEL_TYPE_SINGLE, MODEL_TYPE_PIPELINE, MODEL_TYPE_SMART_ROUTER.
    sections : list of dict
        Each dict has 'name' (str) and 'data' (bytes).
    metadata : dict, optional
        Arbitrary JSON-serializable metadata to store in the header.

    Returns
    -------
    str
        The absolute path to the saved file.
    """
    import platform
    from PipelineTS import __version__

    path = str(Path(path).absolute())

    # Build section descriptors (offset/size/checksum filled after serialization)
    section_descriptors = []
    section_blobs = []
    current_offset = 0  # relative to start of section data area

    for sec in sections:
        blob = sec['data']
        desc = {
            'name': sec['name'],
            'offset': current_offset,
            'size': len(blob),
            'checksum': _sha256(blob),
        }
        section_descriptors.append(desc)
        section_blobs.append(blob)
        current_offset += len(blob)

    # Build header
    header = {
        'model_type': model_type,
        'format_version': FORMAT_VERSION,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pipelinets_version': __version__,
        'python_version': platform.python_version(),
        'checksum_algo': 'sha256',
        'sections': section_descriptors,
        'metadata': metadata or {},
    }

    header_bytes = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

    # Assemble the payload (everything that gets checksummed)
    payload = bytearray()
    payload.extend(MAGIC)                                    # 8 bytes
    payload.extend(struct.pack('<H', FORMAT_VERSION))        # 2 bytes
    payload.extend(struct.pack('<I', len(header_bytes)))     # 4 bytes
    payload.extend(header_bytes)                             # variable
    for blob in section_blobs:
        payload.extend(blob)                                 # variable

    # Global checksum over the entire payload
    global_checksum = _sha256_bytes(bytes(payload))          # 32 bytes

    # Write to file atomically (write to tmp then rename)
    import tempfile
    import os

    dir_path = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.pts.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.write(global_checksum)
            f.write(FOOTER_MAGIC)                            # 4 bytes
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return path


def _read_pts(path, verify_checksum=True):
    """Read a .pts binary file.

    Parameters
    ----------
    path : str
        Input file path.
    verify_checksum : bool
        If True, verify the global SHA-256 checksum and per-section checksums.

    Returns
    -------
    dict
        Keys: 'header' (dict), 'sections' (dict of name -> bytes).

    Raises
    ------
    ValueError
        If magic number, footer, or checksums don't match.
    """
    path = str(Path(path).absolute())

    with open(path, 'rb') as f:
        data = f.read()

    # Validate footer magic (last 4 bytes)
    if data[-4:] != FOOTER_MAGIC:
        raise ValueError(
            f"Invalid file: footer magic mismatch. "
            f"Expected {FOOTER_MAGIC!r}, got {data[-4:]!r}. File may be corrupted."
        )

    # Validate magic number (first 8 bytes)
    if data[:8] != MAGIC:
        raise ValueError(
            f"Invalid file: magic number mismatch. "
            f"Expected {MAGIC!r}, got {data[:8]!r}. Not a PipelineTS .pts file."
        )

    # Extract global checksum (32 bytes before footer)
    stored_checksum = data[-36:-4]

    # The payload is everything before the checksum + footer
    payload = data[:-36]

    # Verify global checksum
    if verify_checksum:
        computed_checksum = _sha256_bytes(payload)
        if computed_checksum != stored_checksum:
            raise ValueError(
                "Integrity check failed: global SHA-256 checksum mismatch. "
                "The file may have been corrupted or tampered with."
            )

    # Parse format version
    fmt_version = struct.unpack('<H', data[8:10])[0]
    if fmt_version > FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version {fmt_version}. "
            f"This build supports version <= {FORMAT_VERSION}. "
            f"Please upgrade PipelineTS."
        )

    # Parse header length and header JSON
    header_len = struct.unpack('<I', data[10:14])[0]
    header_bytes = data[14:14 + header_len]
    header = json.loads(header_bytes.decode('utf-8'))

    # Extract sections
    section_data_start = 14 + header_len
    sections = {}
    for desc in header['sections']:
        start = section_data_start + desc['offset']
        end = start + desc['size']
        blob = data[start:end]

        # Verify per-section checksum
        if verify_checksum:
            computed = _sha256(blob)
            if computed != desc['checksum']:
                raise ValueError(
                    f"Integrity check failed: section '{desc['name']}' checksum mismatch. "
                    f"Expected {desc['checksum'][:16]}..., got {computed[:16]}..."
                )

        sections[desc['name']] = blob

    return {'header': header, 'sections': sections}


# ─── Save / Load: Single Model ───────────────────────────────────────────────

def _save_single_model_pts(path, model, scaler=None, metadata=None):
    """Save a single model (and optional scaler) to a .pts file.

    Parameters
    ----------
    path : str
        Output file path.
    model : object
        The model to save.
    scaler : object, optional
        Optional scaler to save alongside the model.
    metadata : dict, optional
        Extra metadata to embed in the file header.

    Returns
    -------
    str
        The absolute path to the saved file.
    """
    sections = [{'name': 'model', 'data': _serialize_obj(model)}]

    meta = metadata or {}
    meta['has_scaler'] = scaler is not None

    if scaler is not None:
        sections.append({'name': 'scaler', 'data': _serialize_obj(scaler)})

    return _write_pts(path, MODEL_TYPE_SINGLE, sections, metadata=meta)


def _load_single_model_pts(path, verify_checksum=True):
    """Load a single model from a .pts file.

    Returns
    -------
    object or tuple
        If a scaler was saved, returns (model, scaler). Otherwise returns the model.
    """
    result = _read_pts(path, verify_checksum=verify_checksum)
    header = result['header']
    sections = result['sections']

    model = _deserialize_obj(sections['model'])

    if header.get('metadata', {}).get('has_scaler', False) and 'scaler' in sections:
        scaler = _deserialize_obj(sections['scaler'])
        return model, scaler

    return model


# ─── Save / Load: Pipeline ───────────────────────────────────────────────────

def _save_pipeline_pts(path, pipeline, metadata=None):
    """Save a ModelPipeline to a .pts file.

    Each sub-model is stored as a separate section for granular integrity checks.
    """
    sections = []

    # Serialize pipeline shell (without models)
    pipeline_shell = deepcopy(pipeline)
    pipeline_shell.models_ = []
    pipeline_shell.best_model_ = None
    sections.append({'name': 'pipeline_meta', 'data': _serialize_obj(pipeline_shell)})

    # Serialize each sub-model as a separate section
    model_names = []
    for (sub_model_name, sub_model) in pipeline.models_:
        section_name = f'model:{sub_model_name}'
        sections.append({'name': section_name, 'data': _serialize_obj(sub_model)})
        model_names.append(sub_model_name)

    meta = metadata or {}
    meta['model_names'] = model_names
    meta['n_models'] = len(model_names)

    return _write_pts(path, MODEL_TYPE_PIPELINE, sections, metadata=meta)


def _load_pipeline_pts(path, verify_checksum=True):
    """Load a ModelPipeline from a .pts file."""
    result = _read_pts(path, verify_checksum=verify_checksum)
    header = result['header']
    sections = result['sections']

    # Restore pipeline shell
    pipeline = _deserialize_obj(sections['pipeline_meta'])

    # Restore sub-models
    model_names = header.get('metadata', {}).get('model_names', [])
    models = []
    for name in model_names:
        section_name = f'model:{name}'
        if section_name in sections:
            sub_model = _deserialize_obj(sections[section_name])
            models.append([name, sub_model])

    pipeline.models_ = models

    # Restore best_model_ reference
    if pipeline.leader_board_ is not None and len(pipeline.leader_board_) > 0:
        best_name = pipeline.leader_board_.iloc[0, :]['model']
        for (sub_model_name, sub_model) in pipeline.models_:
            if sub_model_name == best_name:
                pipeline.best_model_ = sub_model
                break

    return pipeline


# ─── Save / Load: SmartRouter ────────────────────────────────────────────────

def _save_smart_router_pts(path, router, metadata=None):
    """Save a SmartRouter to a .pts file.

    The inner pipeline is serialized as a nested .pts blob within a section.
    """
    if router.pipeline_ is None:
        raise ValueError("SmartRouter has not been fitted. Call fit() first.")

    import io as _io

    sections = []

    # Save inner pipeline as a nested .pts blob
    # We serialize it to memory, not to a temp file
    import tempfile
    import os

    fd, tmp_pipeline = tempfile.mkstemp(suffix=FILE_EXTENSION)
    os.close(fd)
    try:
        _save_pipeline_pts(tmp_pipeline, router.pipeline_)
        with open(tmp_pipeline, 'rb') as f:
            pipeline_blob = f.read()
    finally:
        if os.path.exists(tmp_pipeline):
            os.unlink(tmp_pipeline)

    sections.append({'name': 'inner_pipeline', 'data': pipeline_blob})

    # Save router metadata
    router_meta = {
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
    sections.append({'name': 'router_meta', 'data': _serialize_obj(router_meta)})

    return _write_pts(path, MODEL_TYPE_SMART_ROUTER, sections, metadata=metadata)


def _load_smart_router_pts(path, verify_checksum=True):
    """Load a SmartRouter from a .pts file."""
    import tempfile
    import os

    result = _read_pts(path, verify_checksum=verify_checksum)
    sections = result['sections']

    # Load inner pipeline from nested .pts blob
    pipeline_blob = sections['inner_pipeline']
    fd, tmp_pipeline = tempfile.mkstemp(suffix=FILE_EXTENSION)
    os.close(fd)
    try:
        with open(tmp_pipeline, 'wb') as f:
            f.write(pipeline_blob)
        pipeline = _load_pipeline_pts(tmp_pipeline, verify_checksum=verify_checksum)
    finally:
        if os.path.exists(tmp_pipeline):
            os.unlink(tmp_pipeline)

    # Load router metadata
    meta = _deserialize_obj(sections['router_meta'])

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

    return router


# ─── Legacy .zip support (backward compatibility) ────────────────────────────

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
    import random
    import time

    hash_object = hashlib.md5()
    hash_object.update((string + str(random.random()) + str(time.time())).encode('utf-8'))
    encrypted_string = hash_object.hexdigest()
    return encrypted_string[:12]


def _zip_file(zipfile_fp, *file_fp):
    """Compresses multiple files into a zip archive (legacy)."""
    import zipfile

    if not zipfile_fp.endswith('.zip'):
        raise ValueError("`zipfile_fp` must be a string with the `.zip` suffix")

    with zipfile.ZipFile(zipfile_fp, 'w') as zipf:
        for file in file_fp:
            zipf.write(file, Path(file).name)


def _load_zip_file(zipfile_fp):
    """Extracts a zip file to a temporary directory (legacy)."""
    import zipfile

    if Path(zipfile_fp).is_dir():
        raise ValueError("`zipfile_fp` must be a file name, not a directory.")
    if not zipfile_fp.strip().endswith('.zip'):
        raise ValueError("`zipfile_fp` must be a string with the `.zip` suffix")

    tmp_unzip_fp = str(Path(zipfile_fp).parent.absolute().joinpath(
        f'PIPELINETS_MODEL_{_hash_string(zipfile_fp)}_{int(datetime.datetime.now().timestamp()*1e6)}/'))

    with zipfile.ZipFile(zipfile_fp, 'r') as zip_ref:
        zip_ref.extractall(tmp_unzip_fp)

    return tmp_unzip_fp



def _load_single_model_zip(path, unzip_file_path=None, unzip=True):
    """Load a single model from a legacy .zip file."""
    import os
    import shutil
    import cloudpickle

    if unzip:
        unzip_file_fp = _load_zip_file(path)
    else:
        unzip_file_fp = unzip_file_path

    if not any(i.endswith('.pkl') for i in os.listdir(unzip_file_fp)):
        raise ValueError("Zip file must contain one file with the `.pkl` suffix.")

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



def _load_pipeline_zip(path, unzip_file_path=None, unzip=True):
    """Load a pipeline from a legacy .zip file."""
    import os
    import shutil
    import cloudpickle

    if unzip:
        unzip_file_fp = _load_zip_file(path)
        unzip_file_fp = str(Path(unzip_file_fp))
    else:
        unzip_file_fp = unzip_file_path

    if os.listdir(unzip_file_fp).count('pipeline.pkl') != 1:
        raise ValueError("Zip file must contain one file which be named `pipeline.pkl`")

    pipeline = None
    models = []
    for i in os.listdir(unzip_file_fp):
        if i == 'pipeline.pkl':
            with open(str(Path(unzip_file_fp).joinpath(i)), 'rb') as f:
                pipeline = cloudpickle.load(f)
        else:
            models.append([i[:-4], _load_single_model_zip(str(Path(unzip_file_fp).joinpath(i)))])

    pipeline.models_ = models
    for (sub_model_name, sub_model) in pipeline.models_:
        if sub_model_name == pipeline.leader_board_.iloc[0, :]['model']:
            pipeline.best_model_ = sub_model

    shutil.rmtree(unzip_file_fp)
    return pipeline



def _load_smart_router_zip(path, unzip_file_path=None, unzip=True):
    """Load a SmartRouter from a legacy .zip file."""
    import os
    import shutil
    import cloudpickle

    if unzip:
        unzip_file_fp = _load_zip_file(path)
    else:
        unzip_file_fp = unzip_file_path

    meta_fp = str(Path(unzip_file_fp).joinpath('router_meta.pkl'))
    with open(meta_fp, 'rb') as f:
        meta = cloudpickle.load(f)

    pipeline_zip = str(Path(unzip_file_fp).joinpath('inner_pipeline.zip'))
    pipeline = _load_pipeline_zip(pipeline_zip)

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

    if router.ensemble_ is not None:
        router.ensemble_.pipeline = pipeline

    shutil.rmtree(unzip_file_fp)
    return router


# ─── Utility: File info / verification ────────────────────────────────────────

def get_file_info(path):
    """Read and return metadata from a .pts file without loading model data.

    Parameters
    ----------
    path : str
        Path to a .pts file.

    Returns
    -------
    dict
        File header containing model_type, created_at, version info,
        section descriptors with checksums, and user metadata.

    Raises
    ------
    ValueError
        If the file is not a valid .pts file.

    Examples
    --------
    >>> info = get_file_info('my_model.pts')
    >>> print(info['model_type'])       # 'single_model', 'pipeline', or 'smart_router'
    >>> print(info['created_at'])       # ISO timestamp
    >>> print(info['sections'])         # list of {name, offset, size, checksum}
    """
    path = str(Path(path).absolute())

    with open(path, 'rb') as f:
        data = f.read()

    if data[-4:] != FOOTER_MAGIC:
        raise ValueError("Invalid file: footer magic mismatch.")
    if data[:8] != MAGIC:
        raise ValueError("Invalid file: magic number mismatch.")

    header_len = struct.unpack('<I', data[10:14])[0]
    header_bytes = data[14:14 + header_len]
    header = json.loads(header_bytes.decode('utf-8'))

    # Add file size info
    import os
    header['file_size_bytes'] = os.path.getsize(path)

    return header


def verify_file(path):
    """Verify the integrity of a .pts file.

    Parameters
    ----------
    path : str
        Path to a .pts file.

    Returns
    -------
    dict
        Keys: 'valid' (bool), 'global_checksum_ok' (bool),
        'section_checksums' (dict of name -> bool), 'errors' (list of str).

    Examples
    --------
    >>> result = verify_file('my_model.pts')
    >>> assert result['valid'], result['errors']
    """
    result = {
        'valid': True,
        'global_checksum_ok': True,
        'section_checksums': {},
        'errors': [],
    }

    try:
        _read_pts(path, verify_checksum=True)
    except ValueError as e:
        error_msg = str(e)
        result['valid'] = False
        result['errors'].append(error_msg)
        if 'global SHA-256' in error_msg:
            result['global_checksum_ok'] = False
        if 'section' in error_msg:
            # Try to identify which section failed
            for part in error_msg.split("'"):
                if ':' not in part and len(part) > 0:
                    result['section_checksums'][part] = False
        return result

    # If we get here, all checks passed — fill in section details
    try:
        info = get_file_info(path)
        for sec in info.get('sections', []):
            result['section_checksums'][sec['name']] = True
    except Exception:
        pass

    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def save_model(path, model, scaler=None, metadata=None):
    """Save a machine learning model, pipeline, or SmartRouter.

    Saves to the PipelineTS binary format (.pts).

    Parameters
    ----------
    path : str
        The output file path. Must end with '.pts'.
    model : object
        The fitted model, ModelPipeline, or SmartRouter.
    scaler : object, optional
        The scaler associated with the model (only for single models).
    metadata : dict, optional
        Arbitrary JSON-serializable metadata to embed in the file header.

    Returns
    -------
    str
        The path to the saved file.

    Examples
    --------
    >>> from PipelineTS.io import save_model, load_model
    >>> save_model('my_pipeline.pts', pipeline)
    >>> loaded = load_model('my_pipeline.pts')

    >>> # With metadata
    >>> save_model('model.pts', model, metadata={'author': 'Alice', 'dataset': 'v2'})
    """
    _validate_path(path, for_save=True)

    from PipelineTS.pipeline import ModelPipeline
    from PipelineTS.pipeline.smart_router import SmartRouter

    if isinstance(model, SmartRouter):
        return _save_smart_router_pts(path, model, metadata=metadata)
    elif isinstance(model, ModelPipeline):
        return _save_pipeline_pts(path, model, metadata=metadata)
    else:
        return _save_single_model_pts(path, model, scaler=scaler, metadata=metadata)


def load_model(path, verify_checksum=True):
    """Load a machine learning model, pipeline, or SmartRouter.

    Supports .pts format and legacy .zip files (backward compatible, read-only).

    Parameters
    ----------
    path : str
        The file path. Must end with '.pts' or '.zip' (legacy, read-only).
    verify_checksum : bool, default True
        If True, verify SHA-256 checksums when loading .pts files.
        Set to False to skip integrity verification (faster but less safe).

    Returns
    -------
    object
        The loaded model, ModelPipeline, or SmartRouter.

    Raises
    ------
    ValueError
        If checksum verification fails (file corrupted or tampered with).

    Examples
    --------
    >>> model = load_model('my_pipeline.pts')
    >>> model.predict(n=12)

    >>> # Skip checksum for faster loading (not recommended for untrusted files)
    >>> model = load_model('my_pipeline.pts', verify_checksum=False)

    >>> # Legacy .zip files can still be loaded (read-only)
    >>> model = load_model('old_pipeline.zip')
    """
    import os

    p = Path(path)
    if not p.exists():
        raise ValueError(f"File does not exist: {path}")
    if p.is_dir():
        raise ValueError(f"`path` must be a file name, not a directory: {path}")

    if path.endswith('.zip'):
        # Legacy .zip format
        unzip_file_fp = _load_zip_file(path)
        if 'smart_router.marker' in os.listdir(unzip_file_fp):
            return _load_smart_router_zip(path, unzip_file_path=unzip_file_fp, unzip=False)
        elif 'pipeline.pkl' in os.listdir(unzip_file_fp):
            return _load_pipeline_zip(path, unzip_file_path=unzip_file_fp, unzip=False)
        else:
            return _load_single_model_zip(path, unzip_file_path=unzip_file_fp, unzip=False)

    elif path.endswith(FILE_EXTENSION):
        # New .pts binary format
        result = _read_pts(path, verify_checksum=verify_checksum)
        model_type = result['header']['model_type']

        if model_type == MODEL_TYPE_SMART_ROUTER:
            return _load_smart_router_pts(path, verify_checksum=verify_checksum)
        elif model_type == MODEL_TYPE_PIPELINE:
            return _load_pipeline_pts(path, verify_checksum=verify_checksum)
        elif model_type == MODEL_TYPE_SINGLE:
            return _load_single_model_pts(path, verify_checksum=verify_checksum)
        else:
            raise ValueError(f"Unknown model type in .pts header: {model_type}")
    else:
        raise ValueError(
            f"Unsupported file extension. Expected '{FILE_EXTENSION}' or '.zip', got: {path}"
        )
