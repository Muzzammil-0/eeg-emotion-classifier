# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('templates', 'templates')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('mne')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, collect_dynamic_libs

site_packages = 'C:\\Users\\PROJECTS\\EEG_EMOTION_CLASSIFIER\\.venv\\Lib\\site-packages'

mne_datas, mne_binaries,  mne_hidden = collect_all('mne')
sklearn_datas, sklearn_binaries, sklearn_hidden = collect_all('sklearn')
numpy_datas, numpy_binaries, numpy_hidden = collect_all('numpy')
scipy_datas, scipy_binaries, scipy_hidden = collect_all('scipy')
flask_datas, flask_binaries, flask_hidden = collect_all('flask')
flask_cors_datas, flask_cors_binaries, flask_cors_hidden = collect_all('flask_cors')
joblib_datas, joblib_binaries, joblib_hidden = collect_all('joblib')
xgboost_datas, xgboost_binaries, xgboost_hidden = collect_all('xgboost')
pandas_datas, pandas_binaries, pandas_hidden = collect_all('pandas')

extra_hidden = collect_submodules('sklearn') + collect_submodules('numpy') + collect_submodules('scipy') + collect_submodules('mne')

a = Analysis(
    ['sdf.py'],
    pathex=[site_packages],
    binaries=binaries,
    datas=[('templates', 'templates')] + collect_data_files('mne') + collect_data_files('sklearn')  + collect_data_files('numpy') + collect_data_files('scipy'),
    hiddenimports=collect_submodules('mne') + collect_submodules('sklearn') + collect_submodules('numpy') + collect_submodules('scipy') + ['flask', 'flask_cors', 'joblib', 'xgboost', 'pandas'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sdf',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sdf',
)
