# Remove only the value this environment's activation hook supplied.
if [ -n "${_CUBUTTERFLY_AUTO_PROFILE_ROOT:-}" ] &&
   [ "$_CUBUTTERFLY_AUTO_PROFILE_ROOT" = "${CONDA_PREFIX:-}/share/cuButterfly/hardware" ]; then
    if [ "${CUBUTTERFLY_PROFILE_ROOT:-}" = "$_CUBUTTERFLY_AUTO_PROFILE_ROOT" ]; then
        unset CUBUTTERFLY_PROFILE_ROOT
    fi
    unset _CUBUTTERFLY_AUTO_PROFILE_ROOT
fi
if [ -n "${_CUBUTTERFLY_AUTO_TEMPLATE_ROOT:-}" ]; then
    if [ "${CUBUTTERFLY_TEMPLATE_ROOT:-}" = "$_CUBUTTERFLY_AUTO_TEMPLATE_ROOT" ]; then
        unset CUBUTTERFLY_TEMPLATE_ROOT
    fi
    unset _CUBUTTERFLY_AUTO_TEMPLATE_ROOT
fi
if [ -n "${_CUBUTTERFLY_AUTO_MATHDX_ROOT:-}" ]; then
    if [ "${CUBUTTERFLY_MATHDX_ROOT:-}" = "$_CUBUTTERFLY_AUTO_MATHDX_ROOT" ]; then
        unset CUBUTTERFLY_MATHDX_ROOT
    fi
    unset _CUBUTTERFLY_AUTO_MATHDX_ROOT
fi
