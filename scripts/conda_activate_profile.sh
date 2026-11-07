# Sourced by Conda. An explicitly set value (including empty) takes precedence.
if [ "${CUBUTTERFLY_PROFILE_ROOT+x}" != x ] && [ -n "${CONDA_PREFIX:-}" ]; then
    CUBUTTERFLY_PROFILE_ROOT="$CONDA_PREFIX/share/cuButterfly/hardware"
    export CUBUTTERFLY_PROFILE_ROOT
    _CUBUTTERFLY_AUTO_PROFILE_ROOT="$CUBUTTERFLY_PROFILE_ROOT"
fi
if [ "${CUBUTTERFLY_TEMPLATE_ROOT+x}" != x ] &&
   [ -d "${CONDA_PREFIX:-}/share/cuButterfly/templates" ]; then
    CUBUTTERFLY_TEMPLATE_ROOT="$CONDA_PREFIX/share/cuButterfly/templates"
    export CUBUTTERFLY_TEMPLATE_ROOT
    _CUBUTTERFLY_AUTO_TEMPLATE_ROOT="$CUBUTTERFLY_TEMPLATE_ROOT"
fi
if [ "${CUBUTTERFLY_MATHDX_ROOT+x}" != x ] &&
   [ -n "${CUBUTTERFLY_TEMPLATE_ROOT:-}" ] &&
   [ -d "$CUBUTTERFLY_TEMPLATE_ROOT/mathdx" ]; then
    CUBUTTERFLY_MATHDX_ROOT="$CUBUTTERFLY_TEMPLATE_ROOT/mathdx"
    export CUBUTTERFLY_MATHDX_ROOT
    _CUBUTTERFLY_AUTO_MATHDX_ROOT="$CUBUTTERFLY_MATHDX_ROOT"
fi
