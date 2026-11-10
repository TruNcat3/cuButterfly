"""Exercise sourced Conda hooks without installing or initializing CUDA."""
from pathlib import Path
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("shell", ["bash", "dash"])
@pytest.mark.parametrize("case", ["automatic", "explicit", "empty", "edited", "repeat"])
def test_conda_profile_lifecycle(tmp_path, shell, case):
    prefix = tmp_path / "conda prefix"
    body = {
        "automatic": '''
            . "$activate"
            test "$CUBUTTERFLY_PROFILE_ROOT" = "$CONDA_PREFIX/share/cuButterfly/hardware"
            test "$(env | sed -n 's/^CUBUTTERFLY_PROFILE_ROOT=//p')" = "$CUBUTTERFLY_PROFILE_ROOT"
            . "$deactivate"
            test "${CUBUTTERFLY_PROFILE_ROOT+x}" != x
            test "${_CUBUTTERFLY_AUTO_PROFILE_ROOT+x}" != x
        ''',
        "explicit": '''
            export CUBUTTERFLY_PROFILE_ROOT=/user/profile
            . "$activate"
            test "$CUBUTTERFLY_PROFILE_ROOT" = /user/profile
            . "$deactivate"
            test "$CUBUTTERFLY_PROFILE_ROOT" = /user/profile
        ''',
        "empty": '''
            export CUBUTTERFLY_PROFILE_ROOT=
            . "$activate"
            . "$deactivate"
            test "${CUBUTTERFLY_PROFILE_ROOT+x}" = x
            test -z "$CUBUTTERFLY_PROFILE_ROOT"
        ''',
        "edited": '''
            . "$activate"
            export CUBUTTERFLY_PROFILE_ROOT=/new/user/profile
            . "$deactivate"
            test "$CUBUTTERFLY_PROFILE_ROOT" = /new/user/profile
            test "${_CUBUTTERFLY_AUTO_PROFILE_ROOT+x}" != x
        ''',
        "repeat": '''
            . "$activate"
            . "$activate"
            . "$deactivate"
            . "$deactivate"
            test "${CUBUTTERFLY_PROFILE_ROOT+x}" != x
        ''',
    }[case]
    subprocess.run([shell, "-eu", "-c", '''
        unset CUBUTTERFLY_PROFILE_ROOT _CUBUTTERFLY_AUTO_PROFILE_ROOT
        CONDA_PREFIX=$1
        activate=$2
        deactivate=$3
    ''' + body, "test-hooks", str(prefix), str(SCRIPTS / "conda_activate_profile.sh"),
        str(SCRIPTS / "conda_deactivate_profile.sh")], check=True)


@pytest.mark.parametrize("shell", ["bash", "dash"])
@pytest.mark.parametrize("override", ["none", "explicit", "edited"])
def test_installed_template_lifecycle(tmp_path, shell, override):
    prefix = tmp_path / "conda prefix"
    (prefix / "share/cuButterfly/templates/mathdx").mkdir(parents=True)
    subprocess.run([shell, "-eu", "-c", '''
        unset CUBUTTERFLY_PROFILE_ROOT CUBUTTERFLY_TEMPLATE_ROOT CUBUTTERFLY_MATHDX_ROOT
        unset _CUBUTTERFLY_AUTO_PROFILE_ROOT _CUBUTTERFLY_AUTO_TEMPLATE_ROOT _CUBUTTERFLY_AUTO_MATHDX_ROOT
        CONDA_PREFIX=$1
        if [ "$4" = explicit ]; then
            export CUBUTTERFLY_TEMPLATE_ROOT=/user/templates CUBUTTERFLY_MATHDX_ROOT=/user/mathdx
        fi
        . "$2"
        if [ "$4" = explicit ]; then
            test "$CUBUTTERFLY_TEMPLATE_ROOT" = /user/templates
            test "$CUBUTTERFLY_MATHDX_ROOT" = /user/mathdx
        else
            test "$CUBUTTERFLY_TEMPLATE_ROOT" = "$CONDA_PREFIX/share/cuButterfly/templates"
            test "$CUBUTTERFLY_MATHDX_ROOT" = "$CUBUTTERFLY_TEMPLATE_ROOT/mathdx"
        fi
        if [ "$4" = edited ]; then
            export CUBUTTERFLY_TEMPLATE_ROOT=/user/templates CUBUTTERFLY_MATHDX_ROOT=/user/mathdx
        fi
        . "$3"
        if [ "$4" = none ]; then
            test "${CUBUTTERFLY_TEMPLATE_ROOT+x}" != x
            test "${CUBUTTERFLY_MATHDX_ROOT+x}" != x
        else
            test "$CUBUTTERFLY_TEMPLATE_ROOT" = /user/templates
            test "$CUBUTTERFLY_MATHDX_ROOT" = /user/mathdx
        fi
    ''', "test-templates", str(prefix), str(SCRIPTS / "conda_activate_profile.sh"),
        str(SCRIPTS / "conda_deactivate_profile.sh"), override], check=True)
