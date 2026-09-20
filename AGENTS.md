# cuButterfly research

- Continue the user's requested work and authorized plan through implementation,
  execution and affected verification. Use the current milestone's deliverables
  and acceptance criteria; a request to continue resumes its unfinished work.
  Routine local implementation and affected verification within that scope can
  proceed without step-by-step confirmation, subject to actual tool permissions.
  Analysis or review requests produce findings unless implementation is also requested.
- Read documentation by task: [unified planner](docs/unified_planner.md) for API,
  lowering and search; [hardware calibration](docs/hardware_profile_install.md) for
  migration and measurement; the current section of
  [refactor progress](docs/framework_refactor_progress.md) for experiment recovery.
  For method changes, consult the relevant APPT26 paper sections (locally available
  at `../APPT26/samplepaper.tex`) and [mapping methodology](docs/hardware_mapping_methodology.md).
  Historical experiment sections are evidence for their named builds.
- On this research host, reuse `/home/wt/yes/envs/cubutterfly` and the existing
  build. Research defaults to `CUBUTTERFLY_COMPILE_MODE=research`; preserve an
  explicitly selected policy. Reinstall or recalibrate when relevant inputs change.
- For execution changes, run affected correctness tests; performance changes also
  need matching performance cases. Documentation-only edits need documentation
  checks rather than a GPU run. Broaden checks
  for failures or changes to shared contracts. Reuse valid checks when their inputs
  are unchanged; execute the authorized full acceptance matrix as its own milestone.
- Identify the target GPU by UUID, model and memory capacity; an index is a local
  selector. Performance calibration requires exclusive use. If it is busy, save
  checkpoints and advance available CPU work without stopping other users' jobs.
  Honor explicit monitoring requests using resumable tasks or bounded waits;
  otherwise report what is ready to resume. Do not silently restart an expired
  waiting budget or repeatedly request an already agreed GPU arrangement.
- Resume measurements only with compatible build, hardware, compile policy and
  timing protocol. Reuse completed trials; record interrupted or unconfirmed work
  separately. Compilation, plan initialization and CPU checking stay outside kernel
  timing. Calibrate the target before its comprehensive baseline comparison.
- Preserve operator semantics and separate the theoretical design space, executable
  lowerings and measured candidates. Record missing lowerings and budget omissions.
  Performance claims identify precision, sizes, batches, hardware and build;
  a selected subset does not satisfy the full-matrix target or prove global optimality.
